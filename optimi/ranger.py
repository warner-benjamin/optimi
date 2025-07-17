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

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

from optimi.optimizer import OptimiOptimizer
from optimi.utils import HAS_TRITON, _default_to_triton, _device_guard, _get_triton_block_size, debias, debias_beta

__all__ = ["Ranger", "ranger"]


class Ranger(OptimiOptimizer):
    """Ranger optimizer. RAdam with Lookahead.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate
        betas: Coefficients for gradient and squared gradient moving averages (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient. If `decouple_wd` and `decouple_lr` are False,
            applies L2 penalty (default: 0)
        eps: Added to denominator to improve numerical stability (default: 1e-6)
        k: Lookahead synchronization period (default: 6)
        alpha: Lookahead weight interpolation coefficient (default: 0.5)
        decouple_wd: Apply decoupled weight decay instead of L2 penalty (default: True)
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
        k: int = 6,
        alpha: float = 0.5,
        decouple_wd: bool = True,
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
            k=k,
            alpha=alpha,
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
            state["la_param"] = param.data.clone()

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
        la_params: list[Tensor],
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
            la_params.append(state["la_param"])
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
                params, grads, exp_avgs, exp_avg_sqs, la_params, kahan_comps = [], [], [], [], [], []
                self._init_group(group, params, grads, exp_avgs, exp_avg_sqs, la_params, kahan_comps)

                ranger(
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    la_params=la_params,
                    kahan_comps=kahan_comps,
                    lr=group["lr"],
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    k=group["k"],
                    alpha=group["alpha"],
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

            ranger(
                params=param,
                grads=param.grad,
                exp_avgs=state["exp_avg"],
                exp_avg_sqs=state["exp_avg_sq"],
                la_params=state["la_param"],
                kahan_comps=state["kahan_comp"],
                lr=group["lr"],
                beta1=group["beta1"],
                beta2=group["beta2"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                k=group["k"],
                alpha=group["alpha"],
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


def ranger(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    la_params: list[Tensor],
    kahan_comps: list[Tensor | None] | None = None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    k: int,
    alpha: float,
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
    """Functional API to apply a Ranger optimization step.

    See `optimi.Ranger` for more details.

    Args:
        params: Parameters to update
        grads: Parameter gradients
        exp_avgs: Gradient moving averages
        exp_avg_sqs: Squared gradient moving averages
        la_params: Lookahead parameters
        kahan_comps: Kahan summation compensations
        lr: Learning rate
        beta1: Gradient moving average coefficient
        beta2: Squared gradient moving average coefficient
        weight_decay: Weight decay coefficient
        eps: Added to denominator to improve numerical stability
        k: How often to conduct Lookahead step
        alpha: Lookahead weight interpolation coefficient
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

    # compute length of the approximated SMA
    rho_inf = 2 / (1 - beta2) - 1
    rho = rho_inf - 2 * step_int * (beta2**step_int) / debias(beta2, step_int)

    # compute variance rectification term
    if rho > 5:
        rect = math.sqrt(((rho - 4) * (rho - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho))
    else:
        rect = None

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
            func = _single_param_triton_ranger
        elif foreach:
            raise ValueError(f"Gradient release {gradient_release=} and foreach {foreach=} cannot be used together")
        else:
            func = _single_param_ranger
    else:
        if triton:
            func = _triton_ranger
        elif foreach:
            func = _foreach_ranger
        else:
            func = _single_ranger

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        la_params,
        kahan_comps,
        lr=lr,
        beta1_hat=beta1_hat,
        beta1_comp=beta1_comp,
        beta2_hat=beta2_hat,
        beta2_comp=beta2_comp,
        weight_decay=weight_decay,
        eps=eps,
        rect=rect,
        k=k,
        alpha=alpha,
        step=step_int,
        decouple_wd=(decouple_wd or decouple_lr),
        kahan_sum=kahan_sum,
        update_parameters=(not optimizer_accumulation),
    )


def _single_ranger(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    la_params: list[Tensor],
    kahan_comps: list[Tensor | None],
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    beta2_comp: float,
    weight_decay: float,
    eps: float,
    rect: float | None,
    k: int,
    alpha: float,
    step: int,
    decouple_wd: bool,
    kahan_sum: bool = False,
    update_parameters: bool = True,
    **kwargs,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        la_param = la_params[i]
        kahan_comp = kahan_comps[i]

        _single_param_ranger(
            param=param,
            grad=grad,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            la_param=la_param,
            kahan_comp=kahan_comp,
            lr=lr,
            beta1_comp=beta1_comp,
            beta2_hat=beta2_hat,
            beta2_comp=beta2_comp,
            weight_decay=weight_decay,
            eps=eps,
            rect=rect,
            k=k,
            alpha=alpha,
            step=step,
            decouple_wd=decouple_wd,
            kahan_sum=kahan_sum,
            update_parameters=update_parameters,
        )


def _single_param_ranger(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    la_param: Tensor,
    kahan_comp: Tensor | None,
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    beta2_comp: float,
    weight_decay: float,
    eps: float,
    rect: float | None,
    k: int,
    alpha: float,
    step: int,
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
            # RAdam step
            if rect is not None:
                kahan_comp.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr * rect)
            else:
                kahan_comp.add_(exp_avg, alpha=-lr)

            # update weights with kahan compensation using grad as temp buffer
            grad.copy_(param.detach())
            param.add_(kahan_comp)

            # save error back to kahan compensation for next iteration
            kahan_comp.add_(grad.sub_(param))

            # Lookahead step
            if step % k == 0:
                kahan_comp.add_(param.sub_(la_param), alpha=alpha)

                # update weights with kahan compensation using grad as temp buffer
                grad.copy_(la_param.detach())
                la_param.add_(kahan_comp)

                # save error back to kahan compensation for next iteration
                kahan_comp.add_(grad.sub_(la_param), alpha=alpha)

                param.copy_(la_param)
        else:
            # RAdam step
            if rect is not None:
                param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr * rect)
            else:
                param.add_(exp_avg, alpha=-lr)

            # Lookahead step
            if step % k == 0:
                la_param.add_(param.sub(la_param), alpha=alpha)
                param.copy_(la_param)


def _foreach_ranger(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    la_params: list[Tensor],
    kahan_comps: list[Tensor | None],
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    beta2_comp: float,
    weight_decay: float,
    eps: float,
    rect: float | None,
    k: int,
    alpha: float,
    step: int,
    decouple_wd: bool,
    kahan_sum: bool = False,
    **kwargs,
):
    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, la_params, kahan_comps])
    for (_, dtype), ((dev_params, dev_grads, dev_exp_avgs, dev_exp_avg_sqs, dev_la_params, dev_kahan_comps), _) in grouped_tensors.items():
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

        # RAdam denominator using dev_grads as a temp buffer
        if rect is not None:
            torch._foreach_copy_(dev_grads, dev_exp_avg_sqs)
            torch._foreach_sqrt_(dev_grads)
            torch._foreach_add_(dev_grads, eps)

        if do_kahan_sum:
            # RAdam step
            if rect is not None:
                torch._foreach_addcdiv_(dev_kahan_comps, dev_exp_avgs, dev_grads, value=-lr * rect)
            else:
                torch._foreach_add_(dev_kahan_comps, dev_exp_avgs, alpha=-lr)

            # update weights with kahan compensation using dev_grads as temp buffer
            torch._foreach_copy_(dev_grads, dev_params)
            torch._foreach_add_(dev_params, dev_kahan_comps, alpha=1)

            # save error back to kahan compensation for next iteration
            torch._foreach_sub_(dev_grads, dev_params, alpha=1)
            torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=1)

            # Lookahead step
            if step % k == 0:
                torch._foreach_copy_(dev_grads, dev_params)
                torch._foreach_sub_(dev_grads, dev_la_params, alpha=1)
                torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=alpha)

                # update weights with kahan compensation using dev_grads as temp buffer
                torch._foreach_copy_(dev_grads, dev_la_params)
                torch._foreach_add_(dev_la_params, dev_kahan_comps, alpha=1)

                # save error back to kahan compensation for next iteration
                torch._foreach_sub_(dev_grads, dev_la_params, alpha=1)
                torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=1)

                torch._foreach_copy_(dev_params, dev_la_params)
        else:
            # RAdam step
            if rect is not None:
                torch._foreach_addcdiv_(dev_params, dev_exp_avgs, dev_grads, value=-lr * rect)
            else:
                torch._foreach_add_(dev_params, dev_exp_avgs, alpha=-lr)

            # Lookahead step
            if step % k == 0:
                torch._foreach_sub_(dev_params, dev_la_params, alpha=1)
                torch._foreach_add_(dev_la_params, dev_params, alpha=alpha)
                torch._foreach_copy_(dev_params, dev_la_params)


if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _ranger_kernel(
        param_ptr,
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        la_param_ptr,
        kahan_ptr,
        lr: tl.constexpr,
        beta1_hat,
        beta1_comp,
        beta2_hat,
        beta2_comp,
        weight_decay,
        eps,
        rect,
        alpha,
        do_rect: tl.constexpr,
        do_weight_decay: tl.constexpr,
        do_lookahead: tl.constexpr,
        kahan_sum: tl.constexpr,
        decouple_wd: tl.constexpr,
        update_parameters: tl.constexpr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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

        # update gradient moving averages
        exp_avg = tl.fma(exp_avg, beta1_hat, beta1_comp * grad)
        exp_avg_sq = tl.fma(exp_avg_sq, beta2_hat, beta2_comp * grad * grad)

        if update_parameters:
            # RAdam step
            if do_rect:
                step = (-lr * rect) * exp_avg / (tl.sqrt(exp_avg_sq) + eps)
            else:
                step = (-lr) * exp_avg

            if kahan_sum:
                # load kahan compensation, casting to fp32
                kahan_comp = tl.load(kahan_ptr + offsets, mask=mask).to(tl.float32)

                # RAdam step, using the kahan comp instead of param
                kahan_comp = kahan_comp + step

                # update weights with downcasted kahan update
                prev_param = param
                param = param + tl.cast(kahan_comp, param.dtype)

                # save error back to kahan compensation for next iteration
                kahan_comp = kahan_comp + prev_param.to(tl.float32) - param.to(tl.float32)

                # Lookahead update every k steps
                if do_lookahead:
                    la_param = tl.load(la_param_ptr + offsets, mask=mask)

                    # lookahead step
                    la_update = alpha * (param.to(tl.float32) - la_param.to(tl.float32))
                    kahan_comp = kahan_comp + la_update

                    # update slow weights with kahan compensation
                    prev_la_param = la_param
                    la_param = la_param + tl.cast(kahan_comp, la_param.dtype)

                    # save error back to kahan compensation for next iteration
                    kahan_comp = kahan_comp + (prev_la_param.to(tl.float32) - la_param.to(tl.float32))
                    param = la_param  # fast weights sync

                # store kahan compensation
                tl.store(kahan_ptr + offsets, tl.cast(kahan_comp, param.dtype), mask=mask)
            else:
                # Non‑Kahan path
                param = param + tl.cast(step, param.dtype)

                if do_lookahead:
                    la_param = tl.load(la_param_ptr + offsets, mask=mask)
                    la_param = la_param + tl.cast(alpha * (param - la_param), la_param.dtype)
                    param = la_param

            # store parameters
            tl.store(param_ptr + offsets, param, mask=mask)
            if do_lookahead:
                tl.store(la_param_ptr + offsets, la_param, mask=mask)

        # Optionally downcast exp_avg and exp_avg_sq to param.dtype
        tl.store(exp_avg_ptr + offsets, tl.cast(exp_avg, param.dtype), mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, tl.cast(exp_avg_sq, param.dtype), mask=mask)

    def _triton_ranger(
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        la_params: list[Tensor],
        kahan_comps: list[Tensor | None],
        *,
        lr: float,
        beta1_hat: float,
        beta1_comp: float,
        beta2_hat: float,
        beta2_comp: float,
        weight_decay: float,
        eps: float,
        rect: float | None,
        k: int,
        alpha: float,
        step: int,
        decouple_wd: bool,
        kahan_sum: bool,
        **kwargs,
    ) -> None:
        do_lookahead = step % k == 0
        do_weight_decay = weight_decay != 0.0

        if rect is None:
            rect_val: float = 0.0
            do_rect = False
        else:
            rect_val = float(rect)
            do_rect = True

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            la_param = la_params[i]
            kahan_comp = kahan_comps[i]

            n_elements = param.numel()
            block_size = _get_triton_block_size(n_elements)
            grid = (triton.cdiv(n_elements, block_size),)

            # Without this Triton tries to launch from device:0 and we get
            # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
            with _device_guard(param):
                _ranger_kernel[grid](
                    param_ptr=param,
                    grad_ptr=grad,
                    exp_avg_ptr=exp_avg,
                    exp_avg_sq_ptr=exp_avg_sq,
                    la_param_ptr=la_param,
                    kahan_ptr=kahan_comp,
                    lr=lr,
                    beta1_hat=beta1_hat,
                    beta1_comp=beta1_comp,
                    beta2_hat=beta2_hat,
                    beta2_comp=beta2_comp,
                    weight_decay=weight_decay,
                    eps=eps,
                    rect=rect_val,
                    alpha=alpha,
                    do_rect=do_rect,
                    do_weight_decay=do_weight_decay,
                    do_lookahead=do_lookahead,
                    kahan_sum=kahan_sum and param.dtype in [torch.float16, torch.bfloat16],
                    decouple_wd=decouple_wd,
                    update_parameters=True,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
                )

    def _single_param_triton_ranger(
        param: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Tensor,
        la_param: Tensor,
        kahan_comp: Tensor,
        *,
        lr: float,
        beta1_hat: float,
        beta1_comp: float,
        beta2_hat: float,
        beta2_comp: float,
        weight_decay: float,
        eps: float,
        rect: float | None,
        k: int,
        alpha: float,
        step: int,
        decouple_wd: bool,
        kahan_sum: bool,
        update_parameters: bool,
        **kwargs,
    ) -> None:
        n_elements = param.numel()
        block_size = _get_triton_block_size(n_elements)

        grid = (triton.cdiv(n_elements, block_size),)

        if rect is None:
            rect_val: float = 0.0
            do_rect = False
        else:
            rect_val = float(rect)
            do_rect = True

        # Without this Triton tries to launch from device:0 and we get
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        with _device_guard(param):
            _ranger_kernel[grid](
                param_ptr=param,
                grad_ptr=grad,
                exp_avg_ptr=exp_avg,
                exp_avg_sq_ptr=exp_avg_sq,
                la_param_ptr=la_param,
                kahan_ptr=kahan_comp,
                lr=lr,
                beta1_hat=beta1_hat,
                beta1_comp=beta1_comp,
                beta2_hat=beta2_hat,
                beta2_comp=beta2_comp,
                weight_decay=weight_decay,
                eps=eps,
                rect=rect_val,
                alpha=alpha,
                do_rect=do_rect,
                do_weight_decay=weight_decay != 0.0,
                do_lookahead=step % k == 0,
                kahan_sum=kahan_sum and param.dtype in [torch.float16, torch.bfloat16],
                decouple_wd=decouple_wd,
                update_parameters=update_parameters,
                n_elements=n_elements,
                BLOCK_SIZE=block_size,
            )
