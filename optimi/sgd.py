# Copyright (c) 2023 Benjamin Warner
# SPDX-License-Identifier: MIT

# Based on PyTorch Optimizers
# PyTorch - PyTorch BSD-style license - Copyright (c) 2013-present PyTorch contributors

# Based on fastai's SGD implementation
# fastai - Apache License 2.0 - Copyright (c) fast.ai

# Kahan summation inspired by Torch Distributed Experimental's `AnyPrecisionAdamW`
# torchdistX - BSD 3-Clause License - Copyright (c) Meta Platforms, Inc. and affiliates

# Learning rate decoupled weight decay inspired by Composer's `DecoupledSGDW` & `DecoupledAdamW`
# Composer - Apache License 2.0 - Copyright (c) 2022 MosaicML Composer authors

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import _default_to_fused_or_foreach
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

from optimi.optimizer import OptimiOptimizer

__all__ = ["SGD", "sgd"]


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
        foreach: Enables the foreach implementation. If unspecified, tries to use foreach over
            for-loop implementation since it is significantly faster (default: None)
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
        foreach: bool | None = None,
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
            torch_init=torch_init,
            foreach=foreach,
            gradient_release=gradient_release,
            setup=False,
        )
        super().__init__(params, defaults)

    def _init_state(self, group: dict[str, Any], state: dict[Tensor, Any], param: Tensor):
        if "kahan_comp" not in state:
            if group["dampening"] and group["torch_init"]:
                state["exp_avg"] = param.grad.detach().clone()
            else:
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)

            if (group["kahan_sum"] or group["kahan_sum"] is None) and param.dtype in [torch.float16, torch.bfloat16]:
                state["kahan_comp"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                group["kahan_sum"] = True
            else:
                state["kahan_comp"] = None

    def _init_group(
        self, group: dict[str, Any], params: list[Tensor], grads: list[Tensor], exp_avgs: list[Tensor], kahan_comps: list[Tensor]
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            params.append(p)
            grads.append(p.grad)
            state = self.state[p]

            self._init_state(group, state, p)

            exp_avgs.append(state["exp_avg"])
            kahan_comps.append(state["kahan_comp"])

        if not group["setup"]:
            group["setup"] = True

            if group["foreach"] is None:
                _, group["foreach"] = _default_to_fused_or_foreach(params, False, False)

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
                params, grads, exp_avgs, kahan_comps = [], [], [], []
                self._init_group(group, params, grads, exp_avgs, kahan_comps)

                sgd(
                    params=params,
                    grads=grads,
                    exp_avgs=exp_avgs,
                    kahan_comps=kahan_comps,
                    lr=group["lr"],
                    momentum=group["momentum"],
                    weight_decay=group["weight_decay"],
                    dampening=group["dampening"],
                    decouple_wd=group["decouple_wd"],
                    decouple_lr=group["decouple_lr"],
                    max_lr=group["max_lr"],
                    kahan_sum=group["kahan_sum"],
                    foreach=group["foreach"],
                    gradient_release=False,
                    optimizer_accumulation=False,
                )
        else:
            state = self.state[param]
            group = state["group"]
            self._init_state(group, state, param)

            sgd(
                params=param,
                grads=param.grad,
                exp_avgs=state["exp_avg"],
                kahan_comps=state["kahan_comp"],
                lr=group["lr"],
                momentum=group["momentum"],
                weight_decay=group["weight_decay"],
                dampening=group["dampening"],
                decouple_wd=group["decouple_wd"],
                decouple_lr=group["decouple_lr"],
                max_lr=group["max_lr"],
                kahan_sum=group["kahan_sum"],
                foreach=False,
                gradient_release=True,
                optimizer_accumulation=self._optimizer_accumulation,
            )

        return loss


def sgd(
    params: list[Tensor] | Tensor,
    grads: list[Tensor] | Tensor,
    exp_avgs: list[Tensor | None] | Tensor | None,
    kahan_comps: list[Tensor | None] | Tensor | None = None,
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    dampening: bool,
    decouple_wd: bool,
    decouple_lr: bool = False,
    max_lr: float | None = None,
    kahan_sum: bool = False,
    foreach: bool = False,
    gradient_release: bool = False,
    optimizer_accumulation: bool = False,
):
    """Functional API to apply a SGD or SGDW optimization step.

    See `optimi.SGD` for more details.

    Args:
        params: Parameter(s) to update
        grads: Paramete(s) gradients
        exp_avgs: Momentum buffer(s)
        kahan_comps: Kahan summation compensation(s)
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: Weight decay coefficient
        dampening: Use dampening for momentum update
        decouple_wd: Apply decoupled weight decay
        decouple_lr: Apply fully decoupled weight decay
        max_lr: Maximum scheduled learning rate for `decouple_lr`
        kahan_sum: Enables Kahan summation for low precision `params`
        foreach: Enables the faster foreach implementation
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

    if foreach:
        func = _foreach_sgd
    elif gradient_release:
        func = _single_param_sgd
    else:
        func = _single_sgd

    func(
        params,
        grads,
        exp_avgs,
        kahan_comps,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        dampening=dampening,
        decouple_wd=(decouple_wd or decouple_lr),
        kahan_sum=kahan_sum,
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


def _foreach_sgd(
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
    **kwargs,
):
    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, exp_avgs, kahan_comps])
    for (_, dtype), ((dev_params, dev_grads, dev_exp_avgs, dev_kahan_comps), _) in grouped_tensors.items():
        # decoupled weight decay, fully decoupled weight decay, or L2 weight decay
        if weight_decay != 0:
            if decouple_wd:
                torch._foreach_mul_(dev_params, scalar=weight_decay)
            else:
                torch._foreach_add_(dev_grads, dev_params, alpha=weight_decay)

        if momentum != 0:
            # SGD with Momentum
            if dampening:
                torch._foreach_lerp_(dev_exp_avgs, dev_grads, weight=1 - momentum)
            else:
                torch._foreach_mul_(dev_exp_avgs, scalar=momentum)
                torch._foreach_add_(dev_exp_avgs, dev_grads, alpha=1)
        else:
            dev_exp_avgs = dev_grads

        if kahan_sum and dtype in [torch.float16, torch.bfloat16]:
            # SGD step (regular step exp_agv = grad)
            torch._foreach_add_(dev_kahan_comps, dev_exp_avgs, alpha=-lr)

            # update weights with kahan compensation using dev_grads as temp buffer
            torch._foreach_copy_(dev_grads, dev_params)
            torch._foreach_add_(dev_params, dev_kahan_comps, alpha=1)

            # save error back to kahan compensation for next iteration
            torch._foreach_sub_(dev_grads, dev_params, alpha=1)
            torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=1)
        else:
            # SGD step (regular step exp_agv = grad)
            torch._foreach_add_(dev_params, dev_exp_avgs, alpha=-lr)
