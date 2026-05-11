# Copyright (c) 2023-present Benjamin Warner
# SPDX-License-Identifier: MIT

# Based on PyTorch Optimizers
# PyTorch - PyTorch BSD-style license - Copyright (c) 2013-present PyTorch contributors

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

from optimi.flash import FlashPrecision, init_flash_state, resolve_effective_flash
from optimi.optimizer import OptimiOptimizer
from optimi.utils import HAS_TRITON, _default_to_triton, _device_guard, _get_triton_block_size, debias_beta

__all__ = ["Adam", "adam"]


class Adam(OptimiOptimizer):
    """Adam optimizer. Optionally with decoupled weight decay (AdamW).

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate
        betas: Coefficients for gradient and squared gradient moving averages (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient. If `decouple_wd` and `decouple_lr` are False,
            applies L2 penalty (default: 0)
        eps: Added to denominator to improve numerical stability (default: 1e-6)
        decouple_wd: Apply decoupled weight decay instead of L2 penalty (default: False)
        decouple_lr: Apply fully decoupled weight decay instead of L2 penalty (default: False)
        max_lr: Maximum scheduled learning rate. Set if `lr` is not the maximum scheduled learning
            rate and `decouple_lr` is True (default: None)
        kahan_sum: Enables Kahan summation for more accurate parameter updates when training in low
            precision (float16 or bfloat16). If unspecified, automatically applies for low precision
            parameters (default: None)
        flash: Enables FlashOptim parameter compensation for low precision Triton training
            (default: None)
        quantize: Reserved for optimizer-state quantization. Not supported yet (default: False)
        stochastic_rounding: Reserved for FlashOptim stochastic rounding. Not supported yet
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
        eps: float = 1e-6,
        decouple_wd: bool = False,
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
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps=}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
            decouple_wd=decouple_wd,
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
        effective_kahan: bool,
    ):
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            if group["gradient_release"]:
                state["step"] = torch.tensor(0, dtype=torch.int32)

        if effective_flash is not None and "flash_comp" not in state:
            state["flash_comp"] = init_flash_state(param, effective_flash)

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
        exp_avg_sqs: list[Tensor],
        kahan_comps: list[Tensor],
        flash_comps: list[Tensor | None],
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
            group["step"] = torch.tensor(0, dtype=torch.int32)

            if group["triton"] is None:
                group["triton"] = _default_to_triton(params)

        effective_flashes = resolve_effective_flash(params, grads, group["flash"], group["triton"], _default_to_triton(params))
        any_effective_flash = any(flash is not None for flash in effective_flashes)
        for param, state, effective_flash in zip(params, states, effective_flashes):
            effective_kahan = (
                not any_effective_flash
                and (group["kahan_sum"] or group["kahan_sum"] is None)
                and param.dtype
                in [
                    torch.float16,
                    torch.bfloat16,
                ]
            )
            self._init_state(group, state, param, effective_flash, effective_kahan)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            kahan_comps.append(state["kahan_comp"])
            flash_comps.append(state["flash_comp"] if effective_flash is not None else None)

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
                params, grads, exp_avgs, exp_avg_sqs, kahan_comps, flash_comps = [], [], [], [], [], []
                self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, kahan_comps, flash_comps)
                if not params:
                    continue
                effective_kahan = any(
                    kahan_comp is not None and kahan_comp.shape == param.shape for kahan_comp, param in zip(kahan_comps, params)
                )

                adam(
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    kahan_comps=kahan_comps,
                    flash_comps=flash_comps,
                    lr=group["lr"],
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    step=group["step"],
                    decouple_wd=group["decouple_wd"],
                    decouple_lr=group["decouple_lr"],
                    max_lr=group["max_lr"],
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
            effective_kahan = (group["kahan_sum"] or group["kahan_sum"] is None) and param.dtype in [torch.float16, torch.bfloat16]
            self._init_state(group, state, param, effective_flash=None, effective_kahan=effective_kahan)

            adam(
                params=param,
                grads=param.grad,
                exp_avgs=state["exp_avg"],
                exp_avg_sqs=state["exp_avg_sq"],
                kahan_comps=state["kahan_comp"],
                flash_comps=None,
                lr=group["lr"],
                beta1=group["beta1"],
                beta2=group["beta2"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                step=state["step"],
                decouple_wd=group["decouple_wd"],
                decouple_lr=group["decouple_lr"],
                max_lr=group["max_lr"],
                kahan_sum=effective_kahan,
                quantize=group["quantize"],
                stochastic_rounding=group["stochastic_rounding"],
                foreach=False,
                triton=group["triton"],
                gradient_release=True,
                optimizer_accumulation=self._optimizer_accumulation,
            )

        return loss


def adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    kahan_comps: list[Tensor | None] | None = None,
    flash_comps: list[Tensor | None] | Tensor | None = None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: Tensor,
    decouple_wd: bool,
    decouple_lr: bool = False,
    max_lr: float | None = None,
    kahan_sum: bool = False,
    quantize: bool = False,
    stochastic_rounding: bool = False,
    foreach: bool = False,
    triton: bool = False,
    gradient_release: bool = False,
    optimizer_accumulation: bool = False,
):
    """Functional API to apply an Adam or AdamW optimization step.

    See `optimi.Adam` for more details.

    Args:
        params: Parameters to update
        grads: Parameter gradients
        exp_avgs: Gradient moving averages
        exp_avg_sqs: Squared gradient moving averages
        kahan_comps: Kahan summation compensations
        flash_comps: FlashOptim parameter compensations
        lr: Learning rate
        beta1: Gradient moving average coefficient
        beta2: Squared gradient moving average coefficient
        weight_decay: Weight decay coefficient
        eps: Added to denominator to improve numerical stability
        step: Step counter used for bias correction
        decouple_wd: Apply decoupled weight decay
        decouple_lr: Apply fully decoupled weight decay
        max_lr: Maximum scheduled learning rate for `decouple_lr`
        kahan_sum: Enables Kahan summation for low precision parameters
        quantize: Reserved for optimizer-state quantization
        stochastic_rounding: Reserved for FlashOptim stochastic rounding
        foreach: Removed; True raises ValueError
        triton: Enables the faster Triton implementation
        gradient_release: Fuses optimizer step as part of the parameter's backward pass
        optimizer_accumulation: Accumulate gradients into state during gradient release step
    """
    # calculate debiased beta hat & complement terms
    step.add_(1)
    step_int = step.item()
    beta1_hat = debias_beta(beta1, step_int)
    beta1_comp = 1 - beta1_hat
    beta2_hat = debias_beta(beta2, step_int)
    beta2_comp = 1 - beta2_hat

    # calculate decoupled weight decay or fully decoupled weight decay
    if weight_decay != 0:
        if decouple_lr:
            weight_decay = 1 - (lr / max_lr) * weight_decay
        elif decouple_wd:
            weight_decay = 1 - lr * weight_decay

    if kahan_comps is None:
        kahan_comps = [None] * len(params)

    if quantize:
        raise ValueError("quantize=True is not supported until the quantization milestone.")
    if stochastic_rounding:
        raise ValueError("stochastic_rounding=True is not supported until the stochastic rounding milestone.")
    if foreach:
        raise ValueError("foreach=True is no longer supported. Set foreach=False or use triton=True.")

    use_flash_impl = False
    if gradient_release:
        if flash_comps is not None:
            raise ValueError("FlashOptim training support does not support gradient_release=True yet.")
        if triton:
            func = _single_param_triton_adam
        else:
            func = _single_param_adam
    else:
        if triton:
            use_flash_impl = flash_comps is not None and any(flash_comp is not None for flash_comp in flash_comps)
            func = _triton_adam_flash if use_flash_impl else _triton_adam
        else:
            if flash_comps is not None and any(flash_comp is not None for flash_comp in flash_comps):
                raise ValueError("FlashOptim training support requires triton=True")
            func = _single_adam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        kahan_comps,
        flash_comps=flash_comps,
        lr=lr,
        beta1_hat=beta1_hat,
        beta1_comp=beta1_comp,
        beta2_hat=beta2_hat,
        beta2_comp=beta2_comp,
        weight_decay=weight_decay,
        eps=eps,
        decouple_wd=(decouple_wd or decouple_lr),
        kahan_sum=kahan_sum,
        update_parameters=(not optimizer_accumulation),
    )


def _single_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    kahan_comps: list[Tensor | None],
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    beta2_comp: float,
    weight_decay: float,
    eps: float,
    decouple_wd: bool,
    kahan_sum: bool = False,
    update_parameters: bool = True,
    **kwargs,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        kahan_comp = kahan_comps[i]

        _single_param_adam(
            param=param,
            grad=grad,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            kahan_comp=kahan_comp,
            lr=lr,
            beta1_comp=beta1_comp,
            beta2_hat=beta2_hat,
            beta2_comp=beta2_comp,
            weight_decay=weight_decay,
            eps=eps,
            decouple_wd=decouple_wd,
            kahan_sum=kahan_sum,
            update_parameters=update_parameters,
        )


def _single_param_adam(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    kahan_comp: Tensor | None,
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    beta2_comp: float,
    weight_decay: float,
    eps: float,
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

    # update gradient moving averages with debiased betas
    exp_avg.lerp_(grad, weight=beta1_comp)
    exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=beta2_comp)

    if update_parameters:
        if kahan_sum and param.dtype in [torch.float16, torch.bfloat16]:
            # Adam step
            kahan_comp.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr)

            # update weights with kahan compensation using grad as temp buffer
            grad.copy_(param.detach())
            param.add_(kahan_comp)

            # save error back to kahan compensation for next iteration
            kahan_comp.add_(grad.sub_(param))
        else:
            # Adam step
            param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr)


if HAS_TRITON:
    import triton
    import triton.language as tl

    from optimi.flash import _apply_flash, _compute_flash

    @triton.jit
    def _adam_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        kahan_ptr,
        lr,
        beta1_hat,
        beta1_comp,
        beta2_hat,
        beta2_comp,
        weight_decay,
        eps,
        do_weight_decay: tl.constexpr,
        kahan_sum: tl.constexpr,
        decouple_wd: tl.constexpr,
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
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask).to(tl.float32)

        # decoupled weight decay, fully decoupled weight decay, or L2 weight decay
        if do_weight_decay and update_parameters:
            if decouple_wd:
                param = tl.cast(param * weight_decay, param.dtype)
            else:
                grad = grad + param.to(tl.float32) * weight_decay

        # update gradient moving averages:
        exp_avg = tl.fma(exp_avg, beta1_hat, beta1_comp * grad)
        exp_avg_sq = tl.fma(exp_avg_sq, beta2_hat, beta2_comp * grad * grad)

        if update_parameters:
            if kahan_sum:
                # load kahan compensation, casting to fp32
                kahan_comp = tl.load(kahan_ptr + offsets, mask=mask).to(tl.float32)

                # AdamW step, using the kahan comp instead of param
                kahan_comp = kahan_comp - (lr * exp_avg / (tl.sqrt(exp_avg_sq) + eps))

                # update weights with downcasted kahan update
                prev_param = param
                param = param + tl.cast(kahan_comp, param.dtype)

                # save error back to kahan compensation for next iteration
                kahan_comp = kahan_comp + prev_param.to(tl.float32) - param.to(tl.float32)

                # store kahan compensation
                tl.store(kahan_ptr + offsets, tl.cast(kahan_comp, param.dtype), mask=mask)
            else:
                # Standard AdamW step, optionally downcasting to param.dtype from fp32 intermediates
                param = param + tl.cast((-lr * exp_avg / (tl.sqrt(exp_avg_sq) + eps)), param.dtype)

            # Store updated parameters
            tl.store(param_ptr + offsets, param, mask=mask)

        # Optionally downcast exp_avg and exp_avg_sq to param.dtype
        tl.store(exp_avg_ptr + offsets, tl.cast(exp_avg, param.dtype), mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, tl.cast(exp_avg_sq, param.dtype), mask=mask)

    def _triton_adam(
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        kahan_comps: list[Tensor | None],
        *,
        lr: float,
        beta1_hat: float,
        beta1_comp: float,
        beta2_hat: float,
        beta2_comp: float,
        weight_decay: float,
        eps: float,
        decouple_wd: bool,
        kahan_sum: bool,
        update_parameters: bool,
        **kwargs,
    ):
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            kahan_comp = kahan_comps[i]

            n_elements = param.numel()
            block_size = _get_triton_block_size(n_elements)
            grid = (triton.cdiv(n_elements, block_size),)

            # Without this Triton always tries to launch from device:0 and we get
            # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
            with _device_guard(param):
                _adam_kernel[grid](
                    param_ptr=param,
                    grad_ptr=grad,
                    exp_avg_ptr=exp_avg,
                    exp_avg_sq_ptr=exp_avg_sq,
                    kahan_ptr=kahan_comp,
                    lr=lr,
                    beta1_hat=beta1_hat,
                    beta1_comp=beta1_comp,
                    beta2_hat=beta2_hat,
                    beta2_comp=beta2_comp,
                    weight_decay=weight_decay,
                    eps=eps,
                    do_weight_decay=weight_decay != 0.0,
                    decouple_wd=decouple_wd,
                    kahan_sum=kahan_sum and param.dtype in [torch.float16, torch.bfloat16],
                    update_parameters=update_parameters,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
                )

    @triton.jit
    def _adam_flash_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        flash_ptr,
        lr,
        beta1_hat,
        beta1_comp,
        beta2_hat,
        beta2_comp,
        weight_decay,
        eps,
        do_weight_decay: tl.constexpr,
        use_flash: tl.constexpr,
        quantize_state: tl.constexpr,
        stochastic_rounding: tl.constexpr,
        decouple_wd: tl.constexpr,
        update_parameters: tl.constexpr,
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
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask).to(tl.float32)

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

        # update gradient moving averages
        exp_avg = tl.fma(exp_avg, beta1_hat, beta1_comp * grad)
        exp_avg_sq = tl.fma(exp_avg_sq, beta2_hat, beta2_comp * grad * grad)

        if update_parameters:
            # Adam step
            param_fp32 = param_fp32 - (lr * exp_avg / (tl.sqrt(exp_avg_sq) + eps))
            param = param_fp32.to(param.dtype)
            tl.store(param_ptr + offsets, param, mask=mask)

            if use_flash:
                flash_comp = _compute_flash(param_fp32, param, flash_t, mantissa_bits, signed_max)
                tl.store(flash_ptr + offsets, flash_comp, mask=mask)

        tl.store(exp_avg_ptr + offsets, tl.cast(exp_avg, param.dtype), mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, tl.cast(exp_avg_sq, param.dtype), mask=mask)

    def _triton_adam_flash(
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        kahan_comps: list[Tensor | None],
        *,
        flash_comps: list[Tensor | None],
        lr: float,
        beta1_hat: float,
        beta1_comp: float,
        beta2_hat: float,
        beta2_comp: float,
        weight_decay: float,
        eps: float,
        decouple_wd: bool,
        kahan_sum: bool,
        update_parameters: bool,
        **kwargs,
    ):
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            flash_comp = flash_comps[i]

            n_elements = param.numel()
            block_size = _get_triton_block_size(n_elements)
            grid = (triton.cdiv(n_elements, block_size),)
            use_flash = flash_comp is not None
            flash_ptr = flash_comp if use_flash else param
            flash_t = tl.int16 if use_flash and flash_comp.dtype == torch.int16 else tl.int8
            signed_max = 32767 if use_flash and flash_comp.dtype == torch.int16 else 127
            mantissa_bits = 10 if param.dtype == torch.float16 else 7

            with _device_guard(param):
                _adam_flash_kernel[grid](
                    param_ptr=param,
                    grad_ptr=grad,
                    exp_avg_ptr=exp_avg,
                    exp_avg_sq_ptr=exp_avg_sq,
                    flash_ptr=flash_ptr,
                    lr=lr,
                    beta1_hat=beta1_hat,
                    beta1_comp=beta1_comp,
                    beta2_hat=beta2_hat,
                    beta2_comp=beta2_comp,
                    weight_decay=weight_decay,
                    eps=eps,
                    do_weight_decay=weight_decay != 0.0,
                    use_flash=use_flash,
                    quantize_state=False,
                    stochastic_rounding=False,
                    decouple_wd=decouple_wd,
                    update_parameters=update_parameters,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
                    mantissa_bits=mantissa_bits,
                    flash_t=flash_t,
                    signed_max=signed_max,
                )

    def _single_param_triton_adam(
        param: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Tensor,
        kahan_comp: Tensor | None,
        *,
        lr: float,
        beta1_hat: float,
        beta1_comp: float,
        beta2_hat: float,
        beta2_comp: float,
        weight_decay: float,
        eps: float,
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
            _adam_kernel[grid](
                param_ptr=param,
                grad_ptr=grad,
                exp_avg_ptr=exp_avg,
                exp_avg_sq_ptr=exp_avg_sq,
                kahan_ptr=kahan_comp,
                lr=lr,
                beta1_hat=beta1_hat,
                beta1_comp=beta1_comp,
                beta2_hat=beta2_hat,
                beta2_comp=beta2_comp,
                weight_decay=weight_decay,
                eps=eps,
                do_weight_decay=weight_decay != 0.0,
                decouple_wd=decouple_wd,
                kahan_sum=kahan_sum and param.dtype in [torch.float16, torch.bfloat16],
                update_parameters=update_parameters,
                n_elements=n_elements,
                BLOCK_SIZE=block_size,
            )
