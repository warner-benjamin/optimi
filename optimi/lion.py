# Copyright (c) 2023-present Benjamin Warner
# SPDX-License-Identifier: MIT

# Based on PyTorch Optimizers
# PyTorch - PyTorch BSD-style license - Copyright (c) 2013-present PyTorch contributors

# Kahan summation inspired by Torch Distributed Experimental's `AnyPrecisionAdamW`
# torchdistX - BSD 3-Clause License - Copyright (c) Meta Platforms, Inc. and affiliates

# Learning rate decoupled weight decay inspired by Composer's `DecoupledLionW` & `DecoupledAdamW`
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
    reject_compressed_state_dict,
    restore_integer_state_,
    validate_quantized_state_support,
)
from optimi.utils import HAS_TRITON, _default_to_triton, _device_guard, _get_triton_block_size

__all__ = ["Lion", "lion"]

_QUANTIZED_STATES = {"exp_avg": ("exp_avg_scales", MOMENTUM_STATE)}


class Lion(OptimiOptimizer):
    """Lion optimizer. Evolved Sign Momentum.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate
        betas: Coefficients for update moving average and gradient moving average
            (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient. If `decouple_lr` is False, applies decoupled
            weight decay (default: 0)
        decouple_lr: Apply fully decoupled weight decay instead of decoupled weight decay
            (default: False)
        max_lr: Maximum scheduled learning rate. Set if `lr` is not the maximum scheduled learning
            rate and `decouple_lr` is True (default: None)
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
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0,
        decouple_lr: bool = False,
        max_lr: float | None = None,
        kahan_sum: bool | None = None,
        flash: bool | str | FlashPrecision | None = None,
        quantize: bool = False,
        stochastic_rounding: bool = False,
        foreach: bool | None = None,
        triton: bool | None = None,
        gradient_release: bool = False,
    ):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]=}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]=}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            decouple_lr=decouple_lr,
            max_lr=max_lr,
            kahan_sum=kahan_sum,
            flash=flash,
            quantize=quantize,
            stochastic_rounding=stochastic_rounding,
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
            if effective_quantize:
                state["exp_avg"], state["exp_avg_scales"] = init_quantized_state(param, MOMENTUM_STATE)
            else:
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
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
                state["kahan_comp"] = torch.zeros(1, dtype=param.dtype, device=param.device)
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
            effective_quantize = bool(group["quantize"])
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

                lion(
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_scales=exp_avg_scales,
                    kahan_comps=kahan_comps,
                    flash_comps=flash_comps,
                    stochastic_seeds=stochastic_seeds,
                    lr=group["lr"],
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    weight_decay=group["weight_decay"],
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
            effective_quantize = bool(group["quantize"])
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

            lion(
                params=param,
                grads=param.grad,
                exp_avgs=state["exp_avg"],
                exp_avg_scales=state.get("exp_avg_scales"),
                kahan_comps=state["kahan_comp"],
                flash_comps=state["flash_comp"] if effective_flash is not None else None,
                stochastic_seeds=state.get("stochastic_seed") if effective_flash is not None else None,
                lr=group["lr"],
                beta1=group["beta1"],
                beta2=group["beta2"],
                weight_decay=group["weight_decay"],
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


def lion(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor | None],
    kahan_comps: list[Tensor | None] | None = None,
    flash_comps: list[Tensor | None] | Tensor | None = None,
    stochastic_seeds: list[Tensor | None] | Tensor | None = None,
    exp_avg_scales: list[Tensor | None] | Tensor | None = None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
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
    """Functional API to apply a Lion optimization step.

    See `optimi.Lion` for more details.

    Args:
        params: Parameters to update
        grads: Parameter gradients
        exp_avgs: Gradient moving averages
        exp_avg_scales: Quantized gradient moving average scales
        kahan_comps: Kahan summation compensations
        flash_comps: FlashOptim parameter compensations
        stochastic_seeds: Per-parameter stochastic rounding seeds
        lr: Learning rate
        beta1: Update moving average coefficient
        beta2: Gradient moving average coefficient
        weight_decay: Weight decay coefficient
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
        else:
            weight_decay = 1 - lr * weight_decay

    # beta complement terms
    beta1_comp = 1 - beta1
    beta2_comp = 1 - beta2

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
            func = _single_param_triton_lion_flash if use_flash_impl else _single_param_triton_lion
        else:
            func = _single_param_lion
    else:
        if triton:
            use_flash_impl = quantize or (flash_comps is not None and any(flash_comp is not None for flash_comp in flash_comps))
            func = _triton_lion_flash if use_flash_impl else _triton_lion
        else:
            if quantize:
                raise ValueError("FlashOptim training support requires triton=True")
            if flash_comps is not None and any(flash_comp is not None for flash_comp in flash_comps):
                raise ValueError("FlashOptim training support requires triton=True")
            func = _single_lion

    func(
        params,
        grads,
        exp_avgs,
        kahan_comps,
        exp_avg_scales=exp_avg_scales,
        flash_comps=flash_comps,
        stochastic_seeds=stochastic_seeds,
        lr=lr,
        beta1_comp=beta1_comp,
        beta2_comp=beta2_comp,
        weight_decay=weight_decay,
        kahan_sum=kahan_sum,
        quantize_state=quantize,
        stochastic_rounding=stochastic_rounding,
        step_int=step_int,
        update_parameters=(not optimizer_accumulation),
    )


def _single_lion(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor | None],
    kahan_comps: list[Tensor | None],
    *,
    lr: float,
    beta1_comp: float,
    beta2_comp: float,
    weight_decay: float,
    kahan_sum: bool = False,
    update_parameters: bool = True,
    **kwargs,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        kahan_comp = kahan_comps[i]

        _single_param_lion(
            param,
            grad,
            exp_avg,
            kahan_comp,
            lr=lr,
            beta1_comp=beta1_comp,
            beta2_comp=beta2_comp,
            weight_decay=weight_decay,
            kahan_sum=kahan_sum,
            update_parameters=update_parameters,
        )


def _single_param_lion(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor | None,
    kahan_comp: Tensor | None,
    *,
    lr: float,
    beta1_comp: float,
    beta2_comp: float,
    weight_decay: float,
    kahan_sum: bool = False,
    update_parameters: bool = True,
    **kwargs,
):
    # decoupled weight decay or fully decoupled weight decay
    if weight_decay != 0 and update_parameters:
        param.mul_(weight_decay)

    # parameter update value
    if update_parameters:
        update = exp_avg.lerp(grad, weight=beta1_comp).sign_()

    # update gradient moving average
    exp_avg.lerp_(grad, weight=beta2_comp)

    if update_parameters:
        if kahan_sum and param.dtype in [torch.float16, torch.bfloat16]:
            # Lion step
            kahan_comp.add_(update, alpha=-lr)

            # update weights with kahan compensation using grad as temp buffer
            grad.copy_(param.detach())
            param.add_(kahan_comp)

            # save error back to kahan compensation for next iteration
            kahan_comp.add_(grad.sub_(param))
        else:
            # Lion step
            param.add_(update, alpha=-lr)


if HAS_TRITON:
    import triton
    import triton.language as tl

    from optimi.flash import _apply_flash, _compute_flash

    @triton.jit
    def _lion_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        kahan_ptr,
        lr,
        beta1_comp,
        beta2_comp,
        weight_decay,
        do_weight_decay: tl.constexpr,
        kahan_sum: tl.constexpr,
        update_parameters: tl.constexpr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load data
        param = tl.load(param_ptr + offsets, mask=mask)
        # For low precision, with or without Kahan summation, we want all
        # computation except for the parameter updates to occur in float32.
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)

        # decoupled weight decay or fully decoupled weight decay
        if do_weight_decay and update_parameters:
            param = tl.cast(param * weight_decay, param.dtype)

        # parameter update value
        tmp = exp_avg + beta1_comp * (grad - exp_avg)
        update = tl.where(tmp > 0, 1.0, tl.where(tmp < 0, -1.0, 0.0))

        # update gradient moving average
        exp_avg = exp_avg + beta2_comp * (grad - exp_avg)

        if update_parameters:
            if kahan_sum:
                # load kahan compensation, casting to fp32
                kahan_comp = tl.load(kahan_ptr + offsets, mask=mask).to(tl.float32)

                # Lion step
                kahan_comp = kahan_comp - (lr * update)

                # update weights with downcasted kahan update
                prev_param = param
                param = param + tl.cast(kahan_comp, param.dtype)

                # save error back to kahan compensation for next iteration
                kahan_comp = kahan_comp + prev_param.to(tl.float32) - param.to(tl.float32)

                # store kahan compensation
                tl.store(kahan_ptr + offsets, tl.cast(kahan_comp, param.dtype), mask=mask)
            else:
                # Standard Lion step
                param = param + tl.cast((-lr * update), param.dtype)

            # Store updated parameters
            tl.store(param_ptr + offsets, param, mask=mask)

        # Optionally downcast exp_avg to param.dtype
        tl.store(exp_avg_ptr + offsets, tl.cast(exp_avg, param.dtype), mask=mask)

    def _triton_lion(
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        kahan_comps: list[Tensor],
        *,
        lr: float,
        beta1_comp: float,
        beta2_comp: float,
        weight_decay: float,
        kahan_sum: bool = False,
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

            # Without this Triton tries to launch from device:0 and we get
            # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
            with _device_guard(param):
                _lion_kernel[grid](
                    param_ptr=param,
                    grad_ptr=grad,
                    exp_avg_ptr=exp_avg,
                    kahan_ptr=kahan_comp,
                    lr=lr,
                    beta1_comp=beta1_comp,
                    beta2_comp=beta2_comp,
                    weight_decay=weight_decay,
                    do_weight_decay=weight_decay != 0.0,
                    kahan_sum=kahan_sum and param.dtype in [torch.float16, torch.bfloat16],
                    update_parameters=update_parameters,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
                )

    @triton.jit
    def _lion_flash_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_scale_ptr,
        flash_ptr,
        stochastic_seed,
        lr,
        beta1_comp,
        beta2_comp,
        weight_decay,
        do_weight_decay: tl.constexpr,
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
        if quantize_state:
            exp_avg_scale = tl.load(exp_avg_scale_ptr + pid)
            exp_avg_q = tl.load(exp_avg_ptr + offsets, mask=mask, other=0).to(tl.float32) / 127.0
            exp_avg = exp_avg_q / (2.0 - tl.abs(exp_avg_q)) * exp_avg_scale
        else:
            exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)

        if use_flash:
            flash_comp = tl.load(flash_ptr + offsets, mask=mask, other=0)
            param_fp32 = _apply_flash(param, flash_comp, mantissa_bits, signed_max)
        else:
            param_fp32 = param.to(tl.float32)

        # decoupled weight decay or fully decoupled weight decay
        if do_weight_decay and update_parameters:
            param_fp32 = param_fp32 * weight_decay

        # parameter update value
        tmp = exp_avg + beta1_comp * (grad - exp_avg)
        update = tl.where(tmp > 0, 1.0, tl.where(tmp < 0, -1.0, 0.0))

        # update gradient moving average
        exp_avg = exp_avg + beta2_comp * (grad - exp_avg)

        if update_parameters:
            # Lion step
            param_fp32 = param_fp32 - (lr * update)
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

    def _triton_lion_flash(
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        kahan_comps: list[Tensor],
        *,
        exp_avg_scales: list[Tensor | None],
        flash_comps: list[Tensor | None],
        stochastic_seeds: list[Tensor | None],
        lr: float,
        beta1_comp: float,
        beta2_comp: float,
        weight_decay: float,
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
                _lion_flash_kernel[grid](
                    param_ptr=param,
                    grad_ptr=grad,
                    exp_avg_ptr=exp_avg,
                    exp_avg_scale_ptr=exp_avg_scale,
                    flash_ptr=flash_ptr,
                    stochastic_seed=stochastic_seed,
                    lr=lr,
                    beta1_comp=beta1_comp,
                    beta2_comp=beta2_comp,
                    weight_decay=weight_decay,
                    do_weight_decay=weight_decay != 0.0,
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

    def _single_param_triton_lion(
        param: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        kahan_comp: Tensor | None,
        *,
        lr: float,
        beta1_comp: float,
        beta2_comp: float,
        weight_decay: float,
        kahan_sum: bool,
        update_parameters: bool,
        **kwargs,
    ):
        n_elements = param.numel()
        block_size = _get_triton_block_size(n_elements)

        grid = (triton.cdiv(n_elements, block_size),)

        # Without this Triton tries to launch from device:0 and we get
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        with _device_guard(param):
            _lion_kernel[grid](
                param_ptr=param,
                grad_ptr=grad,
                exp_avg_ptr=exp_avg,
                kahan_ptr=kahan_comp,
                lr=lr,
                beta1_comp=beta1_comp,
                beta2_comp=beta2_comp,
                weight_decay=weight_decay,
                do_weight_decay=weight_decay != 0.0,
                kahan_sum=kahan_sum and param.dtype in [torch.float16, torch.bfloat16],
                update_parameters=update_parameters,
                n_elements=n_elements,
                BLOCK_SIZE=block_size,
            )

    def _single_param_triton_lion_flash(
        param: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        kahan_comp: Tensor | None,
        *,
        exp_avg_scales: Tensor | None,
        flash_comps: Tensor,
        stochastic_seeds: Tensor | None,
        lr: float,
        beta1_comp: float,
        beta2_comp: float,
        weight_decay: float,
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
            _lion_flash_kernel[grid](
                param_ptr=param,
                grad_ptr=grad,
                exp_avg_ptr=exp_avg,
                exp_avg_scale_ptr=exp_avg_scales if quantize_state else param,
                flash_ptr=flash_ptr,
                stochastic_seed=stochastic_seed,
                lr=lr,
                beta1_comp=beta1_comp,
                beta2_comp=beta2_comp,
                weight_decay=weight_decay,
                do_weight_decay=weight_decay != 0.0,
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
