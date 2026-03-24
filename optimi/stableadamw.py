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
from functools import partial
from typing import Any

import torch
from torch import Tensor
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

from .gradientscaling import OptimizerScalingHelper, ScalingInfo
from .adam import _adam_max_scale_up
from .optimizer import KAHAN_DTYPES, OptimiOptimizerWithGradientScaling
from .utils import (
    HAS_TRITON,
    TORCH_TO_TRITON_DTYPE,
    _default_to_triton,
    _device_guard,
    _get_triton_block_size,
    debias_beta,
)

__all__ = ["StableAdamW", "stableadamw"]


# this is required as Optimizer.load_state_dict casts the state to the param's dtype
def _restore_triton_scratch_state(optim: OptimiOptimizerWithGradientScaling):
    "Restore or create scratch to fp32 after potentially cast to low precision by load_state_dict."
    for group in optim.param_groups:
        if group["triton"]:
            for p in group["params"]:
                state = optim.state[p]
                if "mean_square" in state:
                    state["mean_square"] = state["mean_square"].to(dtype=torch.float32, device=p.device)
                else:
                    state["mean_square"] = torch.zeros(1, dtype=torch.float32, device=p.device)


class StableAdamW(OptimiOptimizerWithGradientScaling):
    """StableAdamW optimizer. An AdamW-Adafactor hybrid with learning rate update clipping.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate
        betas: Coefficients for gradient and squared gradient moving averages (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient. If `decouple_lr` is False, applies decoupled weight
            decay (default: 1e-2)
        eps: Added to denominator to improve numerical stability (default: 1e-6)
        decouple_lr: Apply fully decoupled weight decay instead of decoupled weight decay
            (default: False)
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
        scale_down_hysteresis: When the optimizer scaling factor is updated downwards, then further
            divide the scaling factor by this amount. (default: None)
        scale_up_hysteresis: When the optimizer scaling factor is updated upwards, then further
            divide the scaling factor by this amount. Upwards rescaling must be triggered manually
            by setting group["auto_rescale_on_next_iteration"] to True before an optimizer
            iteration. (default: None)
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 1e-2,
        eps: float = 1e-6,
        decouple_lr: bool = False,
        max_lr: float | None = None,
        kahan_sum: bool | None = None,
        foreach: bool | None = None,
        triton: bool | None = None,
        gradient_release: bool = False,
        scale_down_hysteresis: float | None = None,
        scale_up_hysteresis: float | None = None,
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
            decouple_lr=decouple_lr,
            max_lr=max_lr,
            kahan_sum=kahan_sum,
            foreach=foreach,
            triton=triton,
            gradient_release=gradient_release,
            setup=False,
            **self._gradient_scaling_init_arguments(scale_down_hysteresis=scale_down_hysteresis, scale_up_hysteresis=scale_up_hysteresis),
        )
        super().__init__(params, defaults)

        self.register_load_state_dict_post_hook(_restore_triton_scratch_state)

    def _init_state(self, group: dict[str, Any], state: dict[Tensor, Any], param: Tensor):
        if "kahan_comp" not in state:
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["eps_sq"] = torch.tensor(group["eps"] ** 2, dtype=param.dtype, device=param.device)

            if (group["kahan_sum"] or group["kahan_sum"] is None) and param.dtype in KAHAN_DTYPES:
                state["kahan_comp"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                group["kahan_sum"] = True
            elif group["triton"]:
                state["kahan_comp"] = torch.zeros(1, dtype=torch.uint8, device=param.device)
            else:
                state["kahan_comp"] = None

            if group["triton"]:
                state["mean_square"] = torch.zeros(1, dtype=torch.float32, device=param.device)

            if group["gradient_release"]:
                state["step"] = torch.tensor(0, dtype=torch.int32)

    def _init_group(
        self,
        group: dict[str, Any],
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        eps_sqs: list[Tensor],
        kahan_comps: list[Tensor],
        mean_squares: list[Tensor],
    ):
        if not group["setup"]:
            group["setup"] = True
            self._gradient_scaling_init_group(group)
            group["step"] = torch.tensor(0, dtype=torch.int32)

            if group["triton"] is None and group["foreach"] is None:
                group["triton"] = _default_to_triton(params)

        for p in group["params"]:
            if p.grad is None:
                continue

            params.append(p)
            grads.append(p.grad)
            state = self.state[p]

            self._init_state(group, state, p)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            eps_sqs.append(state["eps_sq"])
            kahan_comps.append(state["kahan_comp"])

            if group["triton"]:
                mean_squares.append(state["mean_square"])

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

        scaling_pre = self._gradient_scaling_make_partial_info()
        if scaling_pre is False:
            return loss  # skip because found_inf

        if param is None:
            for group in self.param_groups:
                params, grads, exp_avgs, exp_avg_sqs, eps_sqs, kahan_comps, mean_squares = [], [], [], [], [], [], []
                self._init_group(
                    group=group,
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    eps_sqs=eps_sqs,
                    kahan_comps=kahan_comps,
                    mean_squares=mean_squares,
                )

                stableadamw(
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    exp_avg_sqs=exp_avg_sqs,
                    eps_sqs=eps_sqs,
                    kahan_comps=kahan_comps,
                    lr=group["lr"],
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    step=group["step"],
                    decouple_lr=group["decouple_lr"],
                    max_lr=group["max_lr"],
                    kahan_sum=group["kahan_sum"],
                    foreach=group["foreach"],
                    triton=group["triton"],
                    gradient_release=False,
                    optimizer_accumulation=False,
                    mean_squares=mean_squares,
                    scaling=self._gradient_scaling_group_info(group, scaling_pre),
                )
        else:
            state = self.state[param]
            group = state["group"]
            self._init_state(group, state, param)

            if group["triton"]:
                stableadamw(
                    params=param,
                    grads=param.grad,
                    exp_avgs=state["exp_avg"],
                    exp_avg_sqs=state["exp_avg_sq"],
                    eps_sqs=state["eps_sq"],
                    kahan_comps=state["kahan_comp"],
                    lr=group["lr"],
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    step=state["step"],
                    decouple_lr=group["decouple_lr"],
                    max_lr=group["max_lr"],
                    kahan_sum=group["kahan_sum"],
                    foreach=False,
                    triton=True,
                    gradient_release=True,
                    optimizer_accumulation=self._optimizer_accumulation,
                    mean_squares=state["mean_square"],
                    scaling=self._gradient_scaling_group_info(group, scaling_pre),
                )
            else:
                stableadamw(
                    params=param,
                    grads=param.grad,
                    exp_avgs=state["exp_avg"],
                    exp_avg_sqs=state["exp_avg_sq"],
                    eps_sqs=state["eps_sq"],
                    kahan_comps=state["kahan_comp"],
                    lr=group["lr"],
                    beta1=group["beta1"],
                    beta2=group["beta2"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    step=state["step"],
                    decouple_lr=group["decouple_lr"],
                    max_lr=group["max_lr"],
                    kahan_sum=group["kahan_sum"],
                    foreach=False,
                    triton=False,
                    gradient_release=True,
                    optimizer_accumulation=self._optimizer_accumulation,
                    scaling=self._gradient_scaling_group_info(group, scaling_pre),
                )

        return loss


def stableadamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    eps_sqs: list[Tensor],
    kahan_comps: list[Tensor | None] | None = None,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    weight_decay: float,
    eps: float,
    step: Tensor,
    decouple_lr: bool = False,
    max_lr: float | None = None,
    kahan_sum: bool = False,
    foreach: bool = False,
    triton: bool = False,
    gradient_release: bool = False,
    optimizer_accumulation: bool = False,
    mean_squares: list[Tensor] | None = None,
    scaling: ScalingInfo | None = None,
):
    """Functional API to apply a StableAdamW optimization step.

    See `optimi.StableAdamW` for more details.

    Args:
        params: Parameters to update
        grads: Parameter gradients
        exp_avgs: Gradient moving averages
        exp_avg_sqs: Squared gradient moving averages
        eps_sqs: Squared epsilon term tensors
        kahan_comps: Kahan summation compensations
        lr: Learning rate
        beta1: Gradient moving average coefficient
        beta2: Squared gradient moving average coefficient
        weight_decay: Weight decay coefficient
        eps: Added to denominator to improve numerical stability
        step: Step counter used for bias correction
        decouple_lr: Apply fully decoupled weight decay
        max_lr: Maximum scheduled learning rate for `decouple_lr`
        kahan_sum: Enables Kahan summation for low precision parameters
        foreach: Enables the faster foreach implementation
        triton: Enables Triton support for the optimizer
        gradient_release: Fuses optimizer step as part of the parameter's backward pass
        optimizer_accumulation: Accumulate gradients into state during gradient release step
        mean_squares: RMS calculation scratch tensor for triton kernel
        scaling: Gradient scaling information
    """

    rescale_factor = None
    rescale_factor_sqr = None

    if scaling:
        scaler = OptimizerScalingHelper(scaling_info=scaling)
        # The state variables are the same as for adam
        scaler.get_max_optimizer_state_scale_up_factor = partial(_adam_max_scale_up, exp_avgs=exp_avgs, exp_avg_sqs=exp_avg_sqs)
        scaler.update()
        grad_to_opt_scale = scaler.gradient_to_optimizer_scale
        rescale_factor = scaler.rescale_factor
        if rescale_factor is not None:
            rescale_factor_sqr = rescale_factor * rescale_factor
    else:
        grad_to_opt_scale = None

    # calculate debiased beta hat & complement terms
    step.add_(1)
    step_int = step.item()
    beta1_hat = debias_beta(beta1, step_int)
    beta1_comp = 1 - beta1_hat
    beta2_hat = debias_beta(beta2, step_int)
    beta2_comp = 1 - beta2_hat

    if grad_to_opt_scale is not None:
        if foreach:
            raise ValueError(f"Gradient scaling and {foreach=} cannot be used together")

        # Scale the variables to prevent both overflow and underflow.
        beta1_comp *= grad_to_opt_scale
        beta2_comp *= grad_to_opt_scale * grad_to_opt_scale / scaler.optimizer_max
        eps = eps * scaler.optimizer_scale
        eps_sq = eps * eps / scaler.optimizer_max
        scale_rms = scaler.optimizer_max**0.5
        scale_grad_for_rms = grad_to_opt_scale / scale_rms
    else:
        eps_sq = eps * eps
        scale_rms = None
        scale_grad_for_rms = None

    if kahan_comps is None:
        kahan_comps = [None] * len(params)

    if gradient_release:
        if triton:
            func = _single_param_triton_stableadamw
        elif foreach:
            raise ValueError(f"Gradient release {gradient_release=} and foreach {foreach=} cannot be used together")
        else:
            func = _single_param_stableadamw
    else:
        if triton:
            func = _triton_stableadamw
        elif foreach:
            func = _foreach_stableadamw
        else:
            func = _single_stableadamw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        eps_sqs,
        kahan_comps,
        lr=lr,
        beta1_hat=beta1_hat,
        beta1_comp=beta1_comp,
        beta2_hat=beta2_hat,
        beta2_comp=beta2_comp,
        weight_decay=weight_decay,
        eps=eps,
        eps_sq_float=eps_sq,
        decouple_lr=decouple_lr,
        max_lr=max_lr,
        kahan_sum=kahan_sum,
        update_parameters=(not optimizer_accumulation),
        mean_squares=mean_squares,
        scale_rms=scale_rms,
        scale_grad_for_rms=scale_grad_for_rms,
        rescale_factor=rescale_factor,
        rescale_factor_sqr=rescale_factor_sqr,
    )


def _single_stableadamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    eps_sqs: list[Tensor],
    kahan_comps: list[Tensor | None],
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    beta2_comp: float,
    weight_decay: float,
    eps: float,
    decouple_lr: bool,
    max_lr: float | None,
    kahan_sum: bool = False,
    update_parameters: bool = True,
    **kwargs,
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        kahan_comp = kahan_comps[i]
        eps_sq = eps_sqs[i]

        _single_param_stableadamw(
            param=param,
            grad=grad,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            eps_sq=eps_sq,
            kahan_comp=kahan_comp,
            lr=lr,
            beta1_comp=beta1_comp,
            beta2_hat=beta2_hat,
            beta2_comp=beta2_comp,
            weight_decay=weight_decay,
            eps=eps,
            decouple_lr=decouple_lr,
            max_lr=max_lr,
            kahan_sum=kahan_sum,
            update_parameters=update_parameters,
        )


def _single_param_stableadamw(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    eps_sq: Tensor,
    kahan_comp: Tensor | None,
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    beta2_comp: float,
    weight_decay: float,
    eps: float,
    decouple_lr: bool,
    max_lr: float | None,
    scale_rms: float | None = None,
    kahan_sum: bool = False,
    update_parameters: bool = True,
    **kwargs,
):
    # update gradient moving averages with debiased betas
    exp_avg.lerp_(grad, weight=beta1_comp)
    exp_avg_sq.mul_(beta2_hat).addcmul_(grad, grad, value=beta2_comp)

    if update_parameters:
        denom = exp_avg_sq.sqrt()
        if scale_rms is not None:
            denom.mul_(scale_rms)
        denom.add_(eps)

        # compute per tensor RMS stabilization term
        rms = grad.pow(2).div_(exp_avg_sq.maximum(eps_sq)).mean().sqrt()

        # calculate RMS stabilized learning rate
        lr = lr / max(1, rms.item())

        # decoupled weight decay or fully decoupled weight decay
        if weight_decay != 0:
            if decouple_lr:
                weight_decay = 1 - (lr / max_lr) * weight_decay
            else:
                weight_decay = 1 - lr * weight_decay
            param.mul_(weight_decay)

        if kahan_sum and param.dtype in KAHAN_DTYPES:
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


def _foreach_stableadamw(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    eps_sqs: list[Tensor],
    kahan_comps: list[Tensor | None],
    *,
    lr: float,
    beta1_comp: float,
    beta2_hat: float,
    beta2_comp: float,
    weight_decay: float,
    eps: float,
    decouple_lr: bool,
    max_lr: float | None,
    kahan_sum: bool = False,
    **kwargs,
):
    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, eps_sqs, kahan_comps])
    for (_, dtype), ((dev_params, dev_grads, dev_exp_avgs, dev_exp_avg_sqs, dev_eps_sqs, dev_kahan_comps), _) in grouped_tensors.items():
        do_kahan_sum = kahan_sum and dtype in KAHAN_DTYPES

        # update gradient moving averages with debiased betas
        torch._foreach_lerp_(dev_exp_avgs, dev_grads, weight=beta1_comp)
        torch._foreach_mul_(dev_exp_avg_sqs, scalar=beta2_hat)
        torch._foreach_addcmul_(dev_exp_avg_sqs, dev_grads, dev_grads, value=beta2_comp)

        # compute per parameter stabilization terms using dev_grads as temp buffer
        max_exp_avg_sqs = torch._foreach_maximum(dev_exp_avg_sqs, other=dev_eps_sqs)
        torch._foreach_pow_(dev_grads, exponent=2)
        torch._foreach_div_(dev_grads, max_exp_avg_sqs)

        # delete local intermediates to potentially save memory
        del max_exp_avg_sqs

        # calculate RMS stabilized learning rates and optionally weight decay
        if weight_decay != 0:
            neg_lrs, new_wds = [], []
            for r in dev_grads:
                neg_lrs.append(-lr / max(1, r.mean().sqrt().item()))
                if decouple_lr:
                    new_wds.append(1 + (neg_lrs[-1] / max_lr) * weight_decay)
                else:
                    new_wds.append(1 + neg_lrs[-1] * weight_decay)

            # decoupled weight decay or fully decoupled weight decay
            torch._foreach_mul_(dev_params, scalars=new_wds)
        else:
            neg_lrs = [-lr / max(1, r.mean().sqrt().item()) for r in dev_grads]

        # Adam denominator using dev_grads as a temp buffer
        torch._foreach_copy_(dev_grads, dev_exp_avg_sqs)
        torch._foreach_sqrt_(dev_grads)
        torch._foreach_add_(dev_grads, eps)

        if do_kahan_sum:
            # Adam step
            torch._foreach_addcdiv_(dev_kahan_comps, dev_exp_avgs, dev_grads, scalars=neg_lrs)

            # update weights with kahan compensation using dev_grads as temp buffer
            torch._foreach_copy_(dev_grads, dev_params)
            torch._foreach_add_(dev_params, dev_kahan_comps, alpha=1)

            # save error back to kahan compensation for next iteration
            torch._foreach_sub_(dev_grads, dev_params, alpha=1)
            torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=1)
        else:
            # Adam step
            torch._foreach_addcdiv_(dev_params, dev_exp_avgs, dev_grads, scalars=neg_lrs)


if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _stableadamw_exp_avg_kernel(
        grad_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        mean_square_ptr,
        eps,
        eps_sq,
        beta1_hat,
        beta1_comp,
        beta2_hat,
        beta2_comp,
        scale_grad_for_rms,  # used with do_scale
        rescale_factor,  # used with do_rescale
        rescale_factor_sqr,  # used with do_rescale
        n_elements,
        do_rescale: tl.constexpr,
        do_scale: tl.constexpr,
        update_parameters: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        param_dtype: tl.constexpr = tl.float32,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # For low precision, with or without Kahan summation, we want all
        # computation except for the parameter updates to occur in float32.
        grad = tl.load(grad_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask).to(tl.float32)

        if do_rescale:
            # in-place rescale
            exp_avg *= rescale_factor
            exp_avg_sq *= rescale_factor_sqr

        exp_avg = tl.fma(exp_avg, beta1_hat, beta1_comp * grad)
        exp_avg_sq = tl.fma(exp_avg_sq, beta2_hat, beta2_comp * grad * grad)

        # partial calculation of per-element stabilisation term
        if update_parameters:
            # We don't need to worry about underflow or underflow here because all the variables below are float32.
            square = tl.where(mask, (grad * grad) / tl.maximum(exp_avg_sq, eps_sq), 0.0)
            block_sum = tl.sum(square, axis=0, dtype=tl.float32) / n_elements
            if do_scale:
                block_sum *= scale_grad_for_rms * scale_grad_for_rms

            # in testing, this atomic_add was faster then storing the results in
            # a temporary buffer then summing in PyTorch or another Triton kernel
            tl.atomic_add(mean_square_ptr, block_sum)

        # Optionally downcast exp_avg and exp_avg_sq to param.dtype
        tl.store(exp_avg_ptr + offsets, tl.cast(exp_avg, param_dtype), mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, tl.cast(exp_avg_sq, param_dtype), mask=mask)

    @triton.jit
    def _stableadamw_update_kernel(
        param_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        kahan_ptr,
        mean_square_ptr,
        lr,
        weight_decay,
        eps,
        max_lr,
        scale_rms,  # used with do_scale
        do_scale: tl.constexpr,
        do_weight_decay: tl.constexpr,
        kahan_sum: tl.constexpr,
        decouple_lr: tl.constexpr,
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
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask).to(tl.float32)

        # RMS stabilized learning rate
        mean_square = tl.load(mean_square_ptr)
        lr = lr / tl.maximum(1.0, tl.sqrt(mean_square))  # TODO: move this out of the kernel

        rms = tl.sqrt(exp_avg_sq)
        if do_scale:
            rms *= scale_rms

        if do_weight_decay:
            if decouple_lr:
                weight_decay = (lr / max_lr) * weight_decay
            else:
                weight_decay = lr * weight_decay

        # AdamW update
        delta = -(lr * exp_avg / (rms + eps))

        if kahan_sum:
            # load kahan compensation, casting to fp32
            kahan_comp = tl.load(kahan_ptr + offsets, mask=mask).to(tl.float32)

            # decoupled weight decay or fully decoupled weight decay
            if do_weight_decay:
                kahan_comp *= 1.0 - weight_decay
                kahan_comp -= param * weight_decay

            # AdamW step, using the kahan comp instead of param
            kahan_comp = kahan_comp + delta

            # update weights with downcasted kahan update
            prev_param = param
            param = param + tl.cast(kahan_comp, param.dtype)

            # save error back to kahan compensation for next iteration
            kahan_comp = kahan_comp + prev_param.to(tl.float32) - param.to(tl.float32)

            # store kahan compensation
            tl.store(kahan_ptr + offsets, tl.cast(kahan_comp, param.dtype), mask=mask)
        else:
            if do_weight_decay:
                delta -= param * weight_decay

            # Standard AdamW step, optionally downcasting to param.dtype from fp32 intermediates
            param = param + tl.cast(delta, param.dtype)

        # Store updated parameters
        tl.store(param_ptr + offsets, param, mask=mask)

    def _triton_stableadamw(
        params: list[Tensor],
        grads: list[Tensor],
        exp_avgs: list[Tensor],
        exp_avg_sqs: list[Tensor],
        eps_sqs: list[Tensor],
        kahan_comps: list[Tensor | None],
        *,
        mean_squares: list[Tensor],
        lr: float,
        beta1_hat: float,
        beta1_comp: float,
        beta2_hat: float,
        beta2_comp: float,
        weight_decay: float,
        eps: float,
        eps_sq_float: float,
        decouple_lr: bool,
        max_lr: float | None = None,
        kahan_sum: bool = False,
        scale_rms: float | None = None,
        scale_grad_for_rms: float | None = None,
        rescale_factor: float | None = None,
        rescale_factor_sqr: float | None = None,
        **kwargs,
    ):
        do_scale = scale_rms is not None

        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            kahan_comp = kahan_comps[i]
            mean_square = mean_squares[i]

            n_elements = param.numel()
            block_size = _get_triton_block_size(n_elements)
            grid = (triton.cdiv(n_elements, block_size),)

            # Without this Triton always tries to launch from device:0 and we get
            # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
            with _device_guard(param):
                _stableadamw_exp_avg_kernel[grid](
                    grad_ptr=grad,
                    exp_avg_ptr=exp_avg,
                    exp_avg_sq_ptr=exp_avg_sq,
                    mean_square_ptr=mean_square,
                    eps=eps,
                    eps_sq=eps_sq_float,
                    beta1_hat=beta1_hat,
                    beta1_comp=beta1_comp,
                    beta2_hat=beta2_hat,
                    beta2_comp=beta2_comp,
                    update_parameters=True,
                    do_scale=do_scale,
                    do_rescale=rescale_factor is not None,
                    rescale_factor=rescale_factor,
                    rescale_factor_sqr=rescale_factor_sqr,
                    scale_grad_for_rms=scale_grad_for_rms,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
                    param_dtype=TORCH_TO_TRITON_DTYPE[param.dtype],
                )

                _stableadamw_update_kernel[grid](
                    param_ptr=param,
                    exp_avg_ptr=exp_avg,
                    exp_avg_sq_ptr=exp_avg_sq,
                    kahan_ptr=kahan_comp,
                    mean_square_ptr=mean_square,
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    max_lr=max_lr,
                    do_scale=do_scale,
                    scale_rms=scale_rms,
                    do_weight_decay=weight_decay != 0.0,
                    kahan_sum=kahan_sum and param.dtype in KAHAN_DTYPES,
                    decouple_lr=decouple_lr,
                    n_elements=n_elements,
                    BLOCK_SIZE=block_size,
                )
                # reset mean_square scratch for next iteration
                mean_square.zero_()

    def _single_param_triton_stableadamw(
        param: Tensor,
        grad: Tensor,
        exp_avg: Tensor,
        exp_avg_sq: Tensor,
        eps_sq: Tensor,
        kahan_comp: Tensor | None,
        *,
        mean_squares: Tensor,
        lr: float,
        beta1_hat: float,
        beta1_comp: float,
        beta2_hat: float,
        beta2_comp: float,
        weight_decay: float,
        eps: float,
        eps_sq_float: float,
        decouple_lr: bool,
        max_lr: float | None = None,
        kahan_sum: bool = False,
        update_parameters: bool = True,
        scale_rms: float | None = None,
        scale_grad_for_rms: float | None = None,
        rescale_factor: float | None = None,
        rescale_factor_sqr: float | None = None,
        **kwargs,
    ):
        do_scale = scale_rms is not None

        n_elements = param.numel()
        block_size = _get_triton_block_size(n_elements)

        grid = (triton.cdiv(n_elements, block_size),)

        # Without this Triton always tries to launch from device:0 and we get
        # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
        with _device_guard(param):
            _stableadamw_exp_avg_kernel[grid](
                grad_ptr=grad,
                exp_avg_ptr=exp_avg,
                exp_avg_sq_ptr=exp_avg_sq,
                mean_square_ptr=mean_squares,
                eps=eps,
                eps_sq=eps_sq_float,
                beta1_hat=beta1_hat,
                beta1_comp=beta1_comp,
                beta2_hat=beta2_hat,
                beta2_comp=beta2_comp,
                update_parameters=update_parameters,
                do_scale=do_scale,
                do_rescale=rescale_factor is not None,
                rescale_factor=rescale_factor,
                rescale_factor_sqr=rescale_factor_sqr,
                scale_grad_for_rms=scale_grad_for_rms,
                n_elements=n_elements,
                BLOCK_SIZE=block_size,
                param_dtype=TORCH_TO_TRITON_DTYPE[param.dtype],
            )

            if update_parameters:
                _stableadamw_update_kernel[grid](
                    param_ptr=param,
                    exp_avg_ptr=exp_avg,
                    exp_avg_sq_ptr=exp_avg_sq,
                    kahan_ptr=kahan_comp,
                    mean_square_ptr=mean_squares,
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    max_lr=max_lr,
                    do_weight_decay=weight_decay != 0.0,
                    kahan_sum=kahan_sum and param.dtype in KAHAN_DTYPES,
                    decouple_lr=decouple_lr,
                    n_elements=n_elements,
                    do_scale=do_scale,
                    scale_rms=scale_rms,
                    BLOCK_SIZE=block_size,
                )
                # reset mean_square scratch for next iteration
                mean_squares.zero_()
