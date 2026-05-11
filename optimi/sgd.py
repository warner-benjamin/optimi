# Copyright (c) 2023-present Benjamin Warner
# SPDX-License-Identifier: MIT

# Based on PyTorch Optimizers
# PyTorch - PyTorch BSD-style license - Copyright (c) 2013-present PyTorch contributors

# Based on fastai's SGD implementation
# fastai - Apache License 2.0 - Copyright (c) fast.ai

# Kahan summation inspired by Torch Distributed Experimental's `AnyPrecisionAdamW`
# torchdistX - BSD 3-Clause License - Copyright (c) Meta Platforms, Inc. and affiliates

# Learning rate decoupled weight decay inspired by Composer's `DecoupledSGDW` & `DecoupledAdamW`
# Composer - Apache License 2.0 - Copyright (c) 2022 MosaicML Composer authors

# Triton kernels inspired by:
# AdamW-Triton-PyTorch - MIT License - Copyright (c) 2024 Less Wright - https://github.com/lessw2020/AdamW-Triton-PyTorch
# lion-pytorch - MIT License - Copyright (c) 2023 Phil Wang - https://github.com/lucidrains/lion-pytorch


from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor

from optimi.flash import (
    FlashPrecision,
    init_flash_state,
    init_stochastic_state,
    resolve_effective_flash,
    validate_stochastic_rounding,
)
from optimi.optimizer import OptimiOptimizer
from optimi.quantization import (
    MOMENTUM_STATE,
    export_quantized_state_dict,
    init_quantized_state,
    quantize_loaded_state_,
    quantize_state_,
    reject_compressed_state_dict,
    restore_integer_state_,
    validate_quantized_state_support,
)
from optimi.utils import HAS_TRITON, _default_to_triton, _device_guard, _get_triton_block_size

__all__ = ["SGD", "sgd"]

_QUANTIZED_STATES = {"exp_avg": ("exp_avg_scales", MOMENTUM_STATE)}


class SGD(OptimiOptimizer):
    """SGD optimizer. Optionally with momentum and decoupled weight decay.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate
        momentum: Momentum factor. Gradient moving average coefficient if `dampening` is True
            (default: 0)
        weight_decay: Weight decay coefficient. If `decouple_wd` and `decouple_lr` are False,
            applies L2 penalty (default: 0)
        dampening: Use dampening for momentum update (default: False)
        decouple_wd: Apply decoupled weight decay instead of L2 penalty (default: False)
        decouple_lr: Apply fully decoupled weight decay instead of L2 penalty (default: False)
        max_lr: Maximum scheduled learning rate. Set if `lr` is not the maximum scheduled learning
            rate and `decouple_lr` is True (default: None)
        torch_init: Initialize momentum buffer with first gradient instead of zeroes. Enable to
            match PyTorch SGD when using dampening (default: False)
        kahan_sum: Enables Kahan summation for more accurate parameter updates when training in low
            precision (float16 or bfloat16). If unspecified, automatically applies for low precision
            parameters (default: None)
        flash: Enables FlashOptim parameter compensation for low precision Triton training
            (default: None)
        quantize: Enables grouped optimizer-state quantization for Triton training
            (default: False)
        stochastic_rounding: Enables stochastic rounding for FlashOptim correction bits
            (default: False)
        foreach: Removed; True raises ValueError
        triton: Enables Triton implementation. If unspecified, tries to use Triton as it is
            significantly faster than the for-loop implementation (default: None)
        gradient_release: Fuses optimizer step and zero_grad as part of the parameter's backward
            pass. Requires model hooks created with `register_gradient_release`. Incompatible with
            closure (default: False)
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float,
        momentum: float = 0,
        weight_decay: float = 0,
        dampening: bool = False,
        decouple_wd: bool = False,
        decouple_lr: bool = False,
        max_lr: float | None = None,
        torch_init: bool = False,
        kahan_sum: bool | None = None,
        flash: bool | str | FlashPrecision | None = None,
        quantize: bool = False,
        stochastic_rounding: bool = False,
        foreach: bool | None = None,
        triton: bool | None = None,
        gradient_release: bool = False,
    ):
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum=}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            decouple_wd=decouple_wd,
            decouple_lr=decouple_lr,
            max_lr=max_lr,
            kahan_sum=kahan_sum,
            flash=flash,
            quantize=quantize,
            stochastic_rounding=stochastic_rounding,
            torch_init=torch_init,
            foreach=foreach,
            triton=triton,
            gradient_release=gradient_release,
            setup=False,
        )
        super().__init__(params, defaults)

    def _init_state(
        self,
        group: dict[str, Any],
        state: dict[Tensor, Any],
        param: Tensor,
        effective_flash: FlashPrecision | None,
        effective_quantize: bool,
        effective_kahan: bool,
    ):
        if "exp_avg" not in state:
            if group["dampening"] and group["torch_init"]:
                exp_avg = param.grad.detach().clone()
                # PyTorch initializes the momentum buffer with the gradient after l2 weight decay
                if group["weight_decay"] != 0 and not group["decouple_lr"] and not group["decouple_wd"]:
                    exp_avg.add_(param, alpha=group["weight_decay"])
                if effective_quantize:
                    state["exp_avg"], state["exp_avg_scales"] = init_quantized_state(param, MOMENTUM_STATE)
                    quantize_state_(exp_avg, state["exp_avg"], state["exp_avg_scales"], MOMENTUM_STATE)
                else:
                    state["exp_avg"] = exp_avg
            elif group["momentum"] != 0:
                if effective_quantize:
                    state["exp_avg"], state["exp_avg_scales"] = init_quantized_state(param, MOMENTUM_STATE)
                else:
                    state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            elif group["triton"]:
                state["exp_avg"] = torch.zeros(1, dtype=param.dtype, device=param.device)
            else:
                state["exp_avg"] = None
            if group["gradient_release"] and group["stochastic_rounding"]:
                state["step"] = torch.tensor(0, dtype=torch.int32)

        if effective_flash is not None and "flash_comp" not in state:
            state["flash_comp"] = init_flash_state(param, effective_flash)
        if effective_flash is not None and group["stochastic_rounding"] and "stochastic_seed" not in state:
            state["stochastic_seed"] = init_stochastic_state(param)

        if "kahan_comp" not in state:
            if effective_kahan:
                state["kahan_comp"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            elif group["triton"]:
                state["kahan_comp"] = torch.zeros(1, dtype=torch.uint8, device=param.device)
            else:
                state["kahan_comp"] = None

    def _init_group(
        self,
        group: dict[str, Any],
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_scales: list[Tensor | None],
        kahan_comps: list[Tensor],
        flash_comps: list[Tensor | None],
        stochastic_seeds: list[Tensor | None],
    ):
        states = []
        for p in group["params"]:
            if p.grad is None:
                continue

            params.append(p)
            grads.append(p.grad)
            states.append(self.state[p])

        if not params:
            return

        if not group["setup"]:
            group["setup"] = True

            if group["triton"] is None:
                group["triton"] = _default_to_triton(params)
            if group["stochastic_rounding"]:
                group["step"] = torch.tensor(0, dtype=torch.int32)

        triton_supported = _default_to_triton(params)
        if group["quantize"]:
            validate_quantized_state_support(params, grads, group["triton"], triton_supported)

        effective_flashes = resolve_effective_flash(
            params,
            grads,
            group["flash"],
            group["triton"],
            triton_supported,
            require_contiguous=group["stochastic_rounding"],
        )
        any_effective_flash = any(flash is not None for flash in effective_flashes)
        if group["stochastic_rounding"] and not any_effective_flash:
            raise ValueError("stochastic_rounding=True requires FlashOptim parameter compensation.")
        for param, state, effective_flash in zip(params, states, effective_flashes):
            effective_quantize = bool(group["quantize"] and group["momentum"] != 0)
            effective_kahan = (
                not any_effective_flash
                and not effective_quantize
                and (group["kahan_sum"] or group["kahan_sum"] is None)
                and param.dtype in [torch.float16, torch.bfloat16]
            )
            self._init_state(group, state, param, effective_flash, effective_quantize, effective_kahan)

            exp_avgs.append(state["exp_avg"])
            exp_avg_scales.append(state.get("exp_avg_scales"))
            kahan_comps.append(state["kahan_comp"])
            flash_comps.append(state["flash_comp"] if effective_flash is not None else None)
            stochastic_seeds.append(state.get("stochastic_seed") if effective_flash is not None else None)

    def state_dict(self):
        """Returns optimizer state with quantized states dequantized."""
        if any(group["quantize"] for group in self.param_groups):
            return export_quantized_state_dict(self, _QUANTIZED_STATES)
        return super().state_dict()

    def load_state_dict(self, state_dict):
        """Loads optimizer state and re-quantizes state for quantized groups."""
        reject_compressed_state_dict(state_dict)
        super().load_state_dict(state_dict)
        restore_integer_state_(self, state_dict, ("flash_comp",))
        quantize_loaded_state_(self, _QUANTIZED_STATES)

    @torch.no_grad()
    def step(self, closure: Callable | None = None, param: Tensor | None = None):
        """Performs a single optimization step on the whole model or individual parameter.

        Args:
            closure: A closure which reevaluates the model and returns the loss. Incompatible with
                performing an optimization step on a single `param`.
            param: An individual parameter to perform a fused optimization step during the backward
                pass. Requires optimizer to be initialized with `gradient_release=True` and model
                hooks created with `register_gradient_release`. Incompatible with `closure`.
        """
        loss = None
        if closure is not None and param is None:
            with torch.enable_grad():
                loss = closure()

        if param is None:
            for group in self.param_groups:
                params, grads, exp_avgs, exp_avg_scales, kahan_comps, flash_comps, stochastic_seeds = [], [], [], [], [], [], []
                self._init_group(group, params, grads, exp_avgs, exp_avg_scales, kahan_comps, flash_comps, stochastic_seeds)
                if not params:
                    continue
                effective_kahan = any(
                    kahan_comp is not None and kahan_comp.shape == param.shape for kahan_comp, param in zip(kahan_comps, params)
                )

                sgd(
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_scales=exp_avg_scales,
                    kahan_comps=kahan_comps,
                    flash_comps=flash_comps,
                    stochastic_seeds=stochastic_seeds,
                    lr=group["lr"],
                    momentum=group["momentum"],
                    weight_decay=group["weight_decay"],
                    dampening=group["dampening"],
                    decouple_wd=group["decouple_wd"],
                    decouple_lr=group["decouple_lr"],
                    max_lr=group["max_lr"],
                    step=group.get("step"),
                    kahan_sum=effective_kahan,
                    quantize=group["quantize"],
                    stochastic_rounding=group["stochastic_rounding"],
                    foreach=group["foreach"],
                    triton=group["triton"],
                    gradient_release=False,
                    optimizer_accumulation=False,
                )
        else:
            state = self.state[param]
            group = state["group"]
            if group["triton"] is None:
                group["triton"] = _default_to_triton([param])
            triton_supported = _default_to_triton([param])
            if group["quantize"]:
                validate_quantized_state_support([param], [param.grad], group["triton"], triton_supported)
            effective_flash = resolve_effective_flash(
                [param],
                [param.grad],
                group["flash"],
                group["triton"],
                triton_supported,
                require_contiguous=group["stochastic_rounding"],
            )[0]
            if group["stochastic_rounding"] and effective_flash is None:
                raise ValueError("stochastic_rounding=True requires FlashOptim parameter compensation.")
            effective_quantize = bool(group["quantize"] and group["momentum"] != 0)
            effective_kahan = (
                effective_flash is None
                and not effective_quantize
                and (group["kahan_sum"] or group["kahan_sum"] is None)
                and param.dtype in [torch.float16, torch.bfloat16]
            )
            self._init_state(
                group,
                state,
                param,
                effective_flash=effective_flash,
                effective_quantize=effective_quantize,
                effective_kahan=effective_kahan,
            )

            sgd(
                params=param,
                grads=param.grad,
                exp_avgs=state["exp_avg"],
                exp_avg_scales=state.get("exp_avg_scales"),
                kahan_comps=state["kahan_comp"],
                flash_comps=state["flash_comp"] if effective_flash is not None else None,
                stochastic_seeds=state.get("stochastic_seed") if effective_flash is not None else None,
                lr=group["lr"],
                momentum=group["momentum"],
                weight_decay=group["weight_decay"],
                dampening=group["dampening"],
                decouple_wd=group["decouple_wd"],
                decouple_lr=group["decouple_lr"],
                max_lr=group["max_lr"],
                step=state.get("step"),
                kahan_sum=effective_kahan,
                quantize=group["quantize"],
                stochastic_rounding=group["stochastic_rounding"],
                foreach=False,
                triton=group["triton"],
                gradient_release=True,
                optimizer_accumulation=self._optimizer_accumulation,
            )

        return loss


def sgd(
    params: list[Tensor] | Tensor,
    grads: list[Tensor] | Tensor,
    exp_avgs: list[Tensor | None] | Tensor | None,
    kahan_comps: list[Tensor | None] | Tensor | None = None,
    flash_comps: list[Tensor | None] | Tensor | None = None,
    stochastic_seeds: list[Tensor | None] | Tensor | None = None,
    exp_avg_scales: list[Tensor | None] | Tensor | None = None,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    dampening: bool,
    decouple_wd: bool,
    decouple_lr: bool = False,
    max_lr: float | None = None,
    step: Tensor | None = None,
    kahan_sum: bool = False,
    quantize: bool = False,
    stochastic_rounding: bool = False,
    foreach: bool = False,
    triton: bool = False,
    gradient_release: bool = False,
    optimizer_accumulation: bool = False,
):
    """Functional API to apply a SGD or SGDW optimization step.

    See `optimi.SGD` for more details.

    Args:
        params: Parameter(s) to update
        grads: Paramete(s) gradients
        exp_avgs: Momentum buffer(s)
        exp_avg_scales: Quantized momentum buffer scales
        kahan_comps: Kahan summation compensation(s)
        flash_comps: FlashOptim parameter compensations
        stochastic_seeds: Per-parameter stochastic rounding seeds
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: Weight decay coefficient
        dampening: Use dampening for momentum update
        decouple_wd: Apply decoupled weight decay
        decouple_lr: Apply fully decoupled weight decay
        max_lr: Maximum scheduled learning rate for `decouple_lr`
        step: Step counter used for stochastic rounding streams
        kahan_sum: Enables Kahan summation for low precision `params`
        quantize: Enables grouped optimizer-state quantization
        stochastic_rounding: Enables stochastic rounding for FlashOptim correction bits
        foreach: Removed; True raises ValueError
        triton: Enables the faster Triton implementation
        gradient_release: Fuses optimizer step as part of the parameter's backward pass
        optimizer_accumulation: Accumulate gradients into state during gradient release step
    """
    # calculate decoupled weight decay or fully decoupled weight decay
    if weight_decay != 0:
        if decouple_lr:
            weight_decay = 1 - (lr / max_lr) * weight_decay
        elif decouple_wd:
            weight_decay = 1 - lr * weight_decay

    if kahan_comps is None:
        kahan_comps = [None] * len(params)

    validate_stochastic_rounding(stochastic_rounding, flash_comps, triton, gradient_release)

    if stochastic_rounding:
        if step is None:
            raise ValueError("stochastic_rounding=True requires a step counter.")
        step.add_(1)
        step_int = step.item()
    else:
        step_int = 0
    if foreach:
        raise ValueError("foreach=True is no longer supported. Set foreach=False or use triton=True.")

    use_flash_impl = False
    if gradient_release:
        use_flash_impl = flash_comps is not None or quantize
        if use_flash_impl and not triton:
            raise ValueError("FlashOptim training support requires triton=True")
        if triton:
            func = _single_param_triton_sgd_flash if use_flash_impl else _single_param_triton_sgd
        else:
            func = _single_param_sgd
    else:
        if triton:
            use_flash_impl = quantize or (flash_comps is not None and any(flash_comp is not None for flash_comp in flash_comps))
            func = _triton_sgd_flash if use_flash_impl else _triton_sgd
        else:
            if quantize:
                raise ValueError("FlashOptim training support requires triton=True")
            if flash_comps is not None and any(flash_comp is not None for flash_comp in flash_comps):
                raise ValueError("FlashOptim training support requires triton=True")
            func = _single_sgd

    func(
        params,
        grads,
        exp_avgs,
        kahan_comps,
        exp_avg_scales=exp_avg_scales,
        flash_comps=flash_comps,
        stochastic_seeds=stochastic_seeds,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        dampening=dampening,
        decouple_wd=(decouple_wd or decouple_lr),
        kahan_sum=kahan_sum,
        quantize_state=quantize and momentum != 0.0,
        stochastic_rounding=stochastic_rounding,
        step_int=step_int,
        update_parameters=(not optimizer_accumulation),
    )


def _single_sgd(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor | None],
    kahan_comps: list[Tensor | None],
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    dampening: bool,
    decouple_wd: bool,
    kahan_sum: bool = False,
    update_parameters: bool = True,
    **kwargs,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        kahan_comp = kahan_comps[i]

        _single_param_sgd(
            param=param,
            grad=grad,
            exp_avg=exp_avg,
            kahan_comp=kahan_comp,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            decouple_wd=decouple_wd,
            kahan_sum=kahan_sum,
            update_parameters=update_parameters,
        )


def _single_param_sgd(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor | None,
    kahan_comp: Tensor | None,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    dampening: bool,
    decouple_wd: bool,
    kahan_sum: bool = False,
    update_parameters: bool = True,
    **kwargs,
):
    # decoupled weight decay, fully decoupled weight decay, or L2 weight decay
    if weight_decay != 0 and update_parameters:
        if decouple_wd:
            param.mul_(weight_decay)
        else:
            grad.add_(param, alpha=weight_decay)

    if momentum != 0:
        # SGD with Momentum
        if dampening:
            exp_avg.lerp_(grad, weight=1 - momentum)
        else:
            exp_avg.mul_(momentum).add_(grad)
    else:
        exp_avg = grad

    if update_parameters:
        if kahan_sum and param.dtype in [torch.float16, torch.bfloat16]:
            # SGD step (regular step exp_agv = grad)
            kahan_comp.add_(exp_avg, alpha=-lr)

            # update weights with kahan compensation using grad as temp buffer
            grad.copy_(param.detach())
            param.add_(kahan_comp)

            # save error back to kahan compensation for next iteration
            kahan_comp.add_(grad.sub_(param))
        else:
            # SGD step (regular step exp_agv = grad)
            param.add_(exp_avg, alpha=-lr)


if HAS_TRITON:
    import triton
    import triton.language as tl

    from optimi.flash import _apply_flash, _compute_flash

    @triton.jit
    def _sgd_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        kahan_ptr,
        lr,
        momentum,
        weight_decay,
        dampening: tl.constexpr,
        decouple_wd: tl.constexpr,
        kahan_sum: tl.constexpr,
        do_weight_decay: tl.constexpr,
        do_momentum: tl.constexpr,
        update_parameters: tl.constexpr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # load data
        param = tl.load(param_ptr + offsets, mask=mask)
        # For low precision, with or without Kahan summation, we want all
        # computation except for the parameter updates to occur in float32.
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)

        # decoupled weight decay, fully decoupled weight decay, or L2 weight decay
        if do_weight_decay and update_parameters:
            if decouple_wd:
                param = tl.cast(param * weight_decay, param.dtype)
            else:
                grad = tl.cast(grad + param.to(tl.float32) * weight_decay, grad.dtype)

        if do_momentum:  # SGD with Momentum
            exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
            if dampening:
                exp_avg = tl.fma(exp_avg, momentum, (1.0 - momentum) * grad)
            else:
                exp_avg = tl.fma(exp_avg, momentum, grad)
        else:
            exp_avg = grad

        if update_parameters:
            if kahan_sum:
                # load kahan compensation, casting to fp32
                kahan_comp = tl.load(kahan_ptr + offsets, mask=mask).to(tl.float32)

                # SGD step (regular step exp_avg = grad)
                kahan_comp = kahan_comp - (lr * exp_avg)

                # update weights with downcasted kahan update
                prev_param = param
                param = param + tl.cast(kahan_comp, param.dtype)

                # save error back to kahan compensation for next iteration
                kahan_comp = kahan_comp + prev_param.to(tl.float32) - param.to(tl.float32)

                # store kahan compensation
                tl.store(kahan_ptr + offsets, tl.cast(kahan_comp, param.dtype), mask=mask)
            else:
                # Standard SGD step, optionally downcasting to param.dtype from fp32 intermediates
                param = param + tl.cast((-lr * exp_avg), param.dtype)

            # Store updated parameters
            tl.store(param_ptr + offsets, param, mask=mask)

        if do_momentum:
            # Optionally downcast exp_avg to param.dtype
            tl.store(exp_avg_ptr + offsets, tl.cast(exp_avg, param.dtype), mask=mask)

    def _triton_sgd(
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        kahan_comps: list[Tensor],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        dampening: bool,
        decouple_wd: bool,
        kahan_sum: bool,
        update_parameters: bool,
        **kwargs,
    ):
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            kahan_comp = kahan_comps[i]

            n_elements = param.numel()
            block_size = _get_triton_block_size(n_elements)
            grid = (triton.cdiv(n_elements, block_size),)

            # Without this Triton always tries to launch from device:0 and we get
            # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
            with _device_guard(param):
                _sgd_kernel[grid](
                    param_ptr=param,
                    grad_ptr=grad,
                    exp_avg_ptr=exp_avg,
                    kahan_ptr=kahan_comp,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    dampening=dampening,
                    decouple_wd=decouple_wd,
                    kahan_sum=kahan_sum and param.dtype in [torch.float16, torch.bfloat16],
                    do_weight_decay=weight_decay != 0.0,
                    do_momentum=momentum != 0.0,
                    update_parameters=update_parameters,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
                )

    @triton.jit
    def _sgd_flash_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_scale_ptr,
        flash_ptr,
        stochastic_seed,
        lr,
        momentum,
        weight_decay,
        dampening: tl.constexpr,
        decouple_wd: tl.constexpr,
        do_weight_decay: tl.constexpr,
        do_momentum: tl.constexpr,
        use_flash: tl.constexpr,
        quantize_state: tl.constexpr,
        stochastic_rounding: tl.constexpr,
        update_parameters: tl.constexpr,
        step_int,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        mantissa_bits: tl.constexpr,
        flash_t: tl.constexpr,
        signed_max: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        param = tl.load(param_ptr + offsets, mask=mask)
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)

        if use_flash:
            flash_comp = tl.load(flash_ptr + offsets, mask=mask, other=0)
            param_fp32 = _apply_flash(param, flash_comp, mantissa_bits, signed_max)
        else:
            param_fp32 = param.to(tl.float32)

        # decoupled weight decay, fully decoupled weight decay, or L2 weight decay
        if do_weight_decay and update_parameters:
            if decouple_wd:
                param_fp32 = param_fp32 * weight_decay
            else:
                grad = grad + param_fp32 * weight_decay

        if do_momentum:
            if quantize_state:
                exp_avg_scale = tl.load(exp_avg_scale_ptr + pid)
                exp_avg_q = tl.load(exp_avg_ptr + offsets, mask=mask, other=0).to(tl.float32) / 127.0
                exp_avg = exp_avg_q / (2.0 - tl.abs(exp_avg_q)) * exp_avg_scale
            else:
                exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
            if dampening:
                exp_avg = tl.fma(exp_avg, momentum, (1.0 - momentum) * grad)
            else:
                exp_avg = tl.fma(exp_avg, momentum, grad)
        else:
            exp_avg = grad

        if update_parameters:
            # SGD step
            param_fp32 = param_fp32 - (lr * exp_avg)
            param = param_fp32.to(param.dtype)
            tl.store(param_ptr + offsets, param, mask=mask)

            if use_flash:
                rng_offsets = offsets + step_int * n_elements
                flash_comp = _compute_flash(
                    param_fp32,
                    param,
                    stochastic_seed,
                    rng_offsets,
                    flash_t,
                    mantissa_bits,
                    signed_max,
                    stochastic_rounding,
                )
                tl.store(flash_ptr + offsets, flash_comp, mask=mask)

        if do_momentum:
            if quantize_state:
                exp_avg_abs = tl.abs(exp_avg)
                exp_avg_scale = tl.maximum(tl.max(tl.where(mask, exp_avg_abs, 0.0), axis=0), 1.0e-12)
                exp_avg_q = exp_avg / exp_avg_scale
                exp_avg_q = 2.0 * exp_avg_q / (1.0 + tl.abs(exp_avg_q))
                exp_avg_q = exp_avg_q * 127.0
                exp_avg_q = tl.where(exp_avg_q >= 0.0, tl.floor(exp_avg_q + 0.5), -tl.floor(-exp_avg_q + 0.5))
                tl.store(exp_avg_ptr + offsets, exp_avg_q.to(tl.int8), mask=mask)
                tl.store(exp_avg_scale_ptr + pid, exp_avg_scale)
            else:
                tl.store(exp_avg_ptr + offsets, tl.cast(exp_avg, param.dtype), mask=mask)

    def _triton_sgd_flash(
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        kahan_comps: list[Tensor],
        *,
        exp_avg_scales: list[Tensor | None],
        flash_comps: list[Tensor | None],
        stochastic_seeds: list[Tensor | None],
        lr: float,
        momentum: float,
        weight_decay: float,
        dampening: bool,
        decouple_wd: bool,
        kahan_sum: bool,
        quantize_state: bool,
        stochastic_rounding: bool,
        step_int: int,
        update_parameters: bool,
        **kwargs,
    ):
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_scale = exp_avg_scales[i] if quantize_state else param
            flash_comp = flash_comps[i]
            stochastic_seed = stochastic_seeds[i].item() if stochastic_rounding else 0

            n_elements = param.numel()
            block_size = 32 if quantize_state else _get_triton_block_size(n_elements)
            grid = (triton.cdiv(n_elements, block_size),)
            use_flash = flash_comp is not None
            flash_ptr = flash_comp if use_flash else param
            flash_t = tl.int16 if use_flash and flash_comp.dtype == torch.int16 else tl.int8
            signed_max = 32767 if use_flash and flash_comp.dtype == torch.int16 else 127
            mantissa_bits = 10 if param.dtype == torch.float16 else 7

            with _device_guard(param):
                _sgd_flash_kernel[grid](
                    param_ptr=param,
                    grad_ptr=grad,
                    exp_avg_ptr=exp_avg,
                    exp_avg_scale_ptr=exp_avg_scale,
                    flash_ptr=flash_ptr,
                    stochastic_seed=stochastic_seed,
                    lr=lr,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    dampening=dampening,
                    decouple_wd=decouple_wd,
                    do_weight_decay=weight_decay != 0.0,
                    do_momentum=momentum != 0.0,
                    use_flash=use_flash,
                    quantize_state=quantize_state,
                    stochastic_rounding=stochastic_rounding,
                    update_parameters=update_parameters,
                    step_int=step_int,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
                    mantissa_bits=mantissa_bits,
                    flash_t=flash_t,
                    signed_max=signed_max,
                )

    def _single_param_triton_sgd(
        param: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        kahan_comp: Tensor,
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        dampening: bool,
        decouple_wd: bool,
        kahan_sum: bool,
        update_parameters: bool,
        **kwargs,
    ):
        n_elements = param.numel()
        block_size = _get_triton_block_size(n_elements)

        grid = (triton.cdiv(n_elements, block_size),)

        # Without this Triton always tries to launch from device:0 and we get
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        with _device_guard(param):
            _sgd_kernel[grid](
                param_ptr=param,
                grad_ptr=grad,
                exp_avg_ptr=exp_avg,
                kahan_ptr=kahan_comp,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                dampening=dampening,
                decouple_wd=decouple_wd,
                kahan_sum=kahan_sum and param.dtype in [torch.float16, torch.bfloat16],
                do_weight_decay=weight_decay != 0.0,
                do_momentum=momentum != 0.0,
                update_parameters=update_parameters,
                n_elements=n_elements,
                BLOCK_SIZE=block_size,
            )

    def _single_param_triton_sgd_flash(
        param: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        kahan_comp: Tensor,
        *,
        exp_avg_scales: Tensor | None,
        flash_comps: Tensor,
        stochastic_seeds: Tensor | None,
        lr: float,
        momentum: float,
        weight_decay: float,
        dampening: bool,
        decouple_wd: bool,
        kahan_sum: bool,
        quantize_state: bool,
        stochastic_rounding: bool,
        step_int: int,
        update_parameters: bool,
        **kwargs,
    ):
        n_elements = param.numel()
        block_size = 32 if quantize_state else _get_triton_block_size(n_elements)
        grid = (triton.cdiv(n_elements, block_size),)
        use_flash = flash_comps is not None
        flash_ptr = flash_comps if use_flash else param
        stochastic_seed = stochastic_seeds.item() if stochastic_rounding else 0
        flash_t = tl.int16 if use_flash and flash_comps.dtype == torch.int16 else tl.int8
        signed_max = 32767 if use_flash and flash_comps.dtype == torch.int16 else 127
        mantissa_bits = 10 if param.dtype == torch.float16 else 7

        with _device_guard(param):
            _sgd_flash_kernel[grid](
                param_ptr=param,
                grad_ptr=grad,
                exp_avg_ptr=exp_avg,
                exp_avg_scale_ptr=exp_avg_scales if quantize_state else param,
                flash_ptr=flash_ptr,
                stochastic_seed=stochastic_seed,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                dampening=dampening,
                decouple_wd=decouple_wd,
                do_weight_decay=weight_decay != 0.0,
                do_momentum=momentum != 0.0,
                use_flash=use_flash,
                quantize_state=quantize_state,
                stochastic_rounding=stochastic_rounding,
                update_parameters=update_parameters,
                step_int=step_int,
                n_elements=n_elements,
                BLOCK_SIZE=block_size,
                mantissa_bits=mantissa_bits,
                flash_t=flash_t,
                signed_max=signed_max,
            )
