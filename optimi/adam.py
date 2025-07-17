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
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

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
        foreach: Enables the foreach implementation. If unspecified, tries to use foreach over
            for-loop implementation since it can be significantly faster (default: None)
        triton: Enables Triton implementation. If unspecified, tries to use Triton as it is
            significantly faster than both for-loop and foreach implementations (default: None)
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
            foreach=foreach,
            triton=triton,
            gradient_release=gradient_release,
            setup=False,
        )
        super().__init__(params, defaults)

    def _init_state(self, group: dict[str, Any], state: dict[Tensor, Any], param: Tensor):
        if "kahan_comp" not in state:
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            if (group["kahan_sum"] or group["kahan_sum"] is None) and param.dtype in [torch.float16, torch.bfloat16]:
                state["kahan_comp"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                group["kahan_sum"] = True
            elif group["triton"]:
                state["kahan_comp"] = torch.zeros(1, dtype=torch.uint8, device=param.device)
            else:
                state["kahan_comp"] = None

            if group["gradient_release"]:
                state["step"] = torch.tensor(0, dtype=torch.int32)

    def _init_group(
        self,
        group: dict[str, Any],
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        kahan_comps: list[Tensor],
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            params.append(p)
            grads.append(p.grad)
            state = self.state[p]

            self._init_state(group, state, p)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            kahan_comps.append(state["kahan_comp"])

        if not group["setup"]:
            group["setup"] = True
            group["step"] = torch.tensor(0, dtype=torch.int32)

            if group["triton"] is None and group["foreach"] is None:
                group["triton"] = _default_to_triton(params)

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
                params, grads, exp_avgs, exp_avg_sqs, kahan_comps = [], [], [], [], []
                self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, kahan_comps)

                adam(
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    kahan_comps=kahan_comps,
                    lr=group["lr"],
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    step=group["step"],
                    decouple_wd=group["decouple_wd"],
                    decouple_lr=group["decouple_lr"],
                    max_lr=group["max_lr"],
                    kahan_sum=group["kahan_sum"],
                    foreach=group["foreach"],
                    triton=group["triton"],
                    gradient_release=False,
                    optimizer_accumulation=False,
                )
        else:
            state = self.state[param]
            group = state["group"]
            self._init_state(group, state, param)

            adam(
                params=param,
                grads=param.grad,
                exp_avgs=state["exp_avg"],
                exp_avg_sqs=state["exp_avg_sq"],
                kahan_comps=state["kahan_comp"],
                lr=group["lr"],
                beta1=group["beta1"],
                beta2=group["beta2"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                step=state["step"],
                decouple_wd=group["decouple_wd"],
                decouple_lr=group["decouple_lr"],
                max_lr=group["max_lr"],
                kahan_sum=group["kahan_sum"],
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
        foreach: Enables the faster foreach implementation
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

    if gradient_release:
        if triton:
            func = _single_param_triton_adam
        elif foreach:
            raise ValueError(f"Gradient release {gradient_release=} and foreach {foreach=} cannot be used together")
        else:
            func = _single_param_adam
    else:
        if triton:
            func = _triton_adam
        elif foreach:
            func = _foreach_adam
        else:
            func = _single_adam

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        kahan_comps,
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


def _foreach_adam(
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
    **kwargs,
):
    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, kahan_comps])
    for (_, dtype), (
        (dev_params, dev_grads, dev_exp_avgs, dev_exp_avg_sqs, dev_kahan_comps),
        _,
    ) in grouped_tensors.items():
        do_kahan_sum = kahan_sum and dtype in [torch.float16, torch.bfloat16]

        # decoupled weight decay, fully decoupled weight decay, or L2 weight decay
        if weight_decay != 0:
            if decouple_wd:
                torch._foreach_mul_(dev_params, scalar=weight_decay)
            else:
                torch._foreach_add_(dev_grads, dev_params, alpha=weight_decay)

        # update gradient moving averages with debiased betas
        torch._foreach_lerp_(dev_exp_avgs, dev_grads, weight=beta1_comp)
        torch._foreach_mul_(dev_exp_avg_sqs, scalar=beta2_hat)
        torch._foreach_addcmul_(dev_exp_avg_sqs, dev_grads, dev_grads, value=beta2_comp)

        # Adam denominator using dev_grads as a temp buffer
        torch._foreach_copy_(dev_grads, dev_exp_avg_sqs)
        torch._foreach_sqrt_(dev_grads)
        torch._foreach_add_(dev_grads, eps)

        if do_kahan_sum:
            # Adam step
            torch._foreach_addcdiv_(dev_kahan_comps, dev_exp_avgs, dev_grads, value=-lr)

            # update weights with kahan compensation using dev_grads as temp buffer
            torch._foreach_copy_(dev_grads, dev_params)
            torch._foreach_add_(dev_params, dev_kahan_comps, alpha=1)

            # save error back to kahan compensation for next iteration
            torch._foreach_sub_(dev_grads, dev_params, alpha=1)
            torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=1)
        else:
            # Adam step
            torch._foreach_addcdiv_(dev_params, dev_exp_avgs, dev_grads, value=-lr)


if HAS_TRITON:
    import triton
    import triton.language as tl

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
                    update_parameters=True,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
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
