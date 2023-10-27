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
from torch.optim.optimizer import Optimizer, _default_to_fused_or_foreach, required
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

__all__ = ["SGD", "sgd"]


class SGD(Optimizer):
    """SGD optimizer. Optionally with momentum and decoupled weight decay.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups
        lr (float): Default learning rate
        momentum (float): Momentum factor. Gradient moving average coefficient if `dampening` is
            True (default: 0)
        weight_decay (float): Weight decay coefficient. If `decouple_wd` and `decouple_lr` are
            False, applies L2 penalty (default: 0)
        dampening (bool): Use dampening for momentum update (default: False)
        decouple_wd (bool): Apply decoupled weight decay instead of L2 penalty (default: False)
        decouple_lr (bool): Apply learning rate decoupled weight decay instead of L2 penalty
            (default: False)
        max_lr (float, optional): Maximum scheduled learning rate. Set if `lr` is not the maximum
            scheduled learning rate and `decouple_lr` is True.
        torch_init (bool): Initialize momentum buffer with first gradient instead of zeroes. Enable
            to match PyTorch SGD when using dampening (default: False)
        kahan_sum (bool, optional): Enables kahan summation for more accurate parameter updates when
            training in low precision (float16 or bfloat16). If unspecified, automatically applies
            for low precision parameters (default: None)
        foreach (bool, optional): Enables the foreach implementation. If unspecified, tries to use
            foreach over for-loop implementation since it is significantly faster (default: None)
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict],
        lr: float = required,  # type: ignore
        momentum: float = 0,
        weight_decay: float = 0,
        dampening: bool = False,
        decouple_wd: bool = False,
        decouple_lr: bool = False,
        max_lr: float | None = None,
        torch_init: bool = False,
        kahan_sum: bool | None = None,
        foreach: bool | None = None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr=}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum=}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay=}")
        if decouple_lr and max_lr is None:
            max_lr = lr

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
            setup=False,
        )
        super().__init__(params, defaults)

    def _init_group(
        self, group: dict[str, Any], params: list[Tensor], grads: list[Tensor], exp_avgs: list[Tensor], kahan_comps: list[Tensor]
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            params.append(p)
            grads.append(p.grad)
            state = self.state[p]

            # State initialization
            if len(state) == 0:
                if group["dampening"] and group["torch_init"]:
                    state["exp_avg"] = p.grad.detach().clone()
                else:
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if (group["kahan_sum"] or group["kahan_sum"] is None) and p.dtype in [torch.float16, torch.bfloat16]:
                    state["kahan_comp"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    group["kahan_sum"] = True
                else:
                    state["kahan_comp"] = None

            exp_avgs.append(state["exp_avg"])
            kahan_comps.append(state["kahan_comp"])

        if not group["setup"]:
            group["setup"] = True

            if group["foreach"] is None:
                _, group["foreach"] = _default_to_fused_or_foreach(params, False, False)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure which reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

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
            )

        return loss


def sgd(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor | None],
    kahan_comps: list[Tensor | None] | None = None,
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
):
    """Functional API to apply a SGD or SGDW optimization step.

    See `optimi.SGD` for more details.

    Args:
        params (list): Parameters to update
        grads (list): Parameter gradients
        exp_avgs (list): Momentum buffers
        kahan_comps (list, optional): Kahan summation compensations
        weight_decay (float): Weight decay coefficient
        lr (float): Learning rate
        momentum (float): Momentum factor
        dampening (bool): Use dampening for momentum update
        decouple_wd (bool): Apply decoupled weight decay
        decouple_lr (bool): Apply learning rate decoupled weight decay
        max_lr (float, optional): Maximum scheduled learning ratefor `decouple_lr`
        kahan_sum (bool): Enables kahan summation for low precision `params`
        foreach (bool): Enables the faster foreach implementation
    """
    # calculate decoupled weight decay or learning rate decoupled weight decay
    if weight_decay != 0:
        if decouple_lr:
            weight_decay = 1 - (lr / max_lr) * weight_decay
        elif decouple_wd:
            weight_decay = 1 - lr * weight_decay

    if kahan_comps is None:
        kahan_comps = [None] * len(params)

    if foreach:
        func = _foreach_sgd
    else:
        func = _single_sgd

    func(
        params=params,
        grads=grads,
        exp_avgs=exp_avgs,
        kahan_comps=kahan_comps,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        dampening=dampening,
        decouple_wd=(decouple_wd or decouple_lr),
        kahan_sum=kahan_sum,
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
):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        kahan_comp = kahan_comps[i]

        # decoupled weight decay, learning rate decoupled weight decay, or L2 weight decay
        if weight_decay != 0:
            if decouple_wd:
                param.mul_(weight_decay)
            else:
                grad.add_(param, alpha=weight_decay)

        if momentum != 0:
            # SGD Momentum
            if dampening:
                exp_avg.lerp_(grad, weight=1 - momentum)
            else:
                exp_avg.mul_(momentum).add_(grad)

            if kahan_sum and param.dtype in [torch.float16, torch.bfloat16]:
                # SGD with Momentum step
                kahan_comp.add_(exp_avg, alpha=-lr)

                # update weights with kahan compensation
                grad.copy_(param.detach())
                param.add_(kahan_comp)

                # save error back to kahan compensation for next iteration
                kahan_comp.add_(grad.sub_(param))

            else:
                # SGD with Momentum step
                param.add_(exp_avg, alpha=-lr)
        else:
            if kahan_sum and param.dtype in [torch.float16, torch.bfloat16]:
                # SGD step
                kahan_comp.add_(grad, alpha=-lr)

                # update weights with kahan compensation using grad as temp buffer
                grad.copy_(param.detach())
                param.add_(kahan_comp)

                # save error back to kahan compensation for next iteration
                kahan_comp.add_(grad.sub_(param))

            else:
                # SGD step
                param.add_(grad, alpha=-lr)


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
):
    grouped_tensors = _group_tensors_by_device_and_dtype([params, grads, exp_avgs, kahan_comps])
    for (_, dtype), ((dev_params, dev_grads, dev_exp_avgs, dev_kahan_comps), _) in grouped_tensors.items():
        # decoupled weight decay, learning rate decoupled weight decay, or L2 weight decay
        if weight_decay != 0:
            if decouple_wd:
                torch._foreach_mul_(dev_params, scalar=weight_decay)
            else:
                torch._foreach_add_(dev_grads, dev_params, alpha=weight_decay)

        if momentum != 0:
            # SGD Momentum
            if dampening:
                torch._foreach_lerp_(dev_exp_avgs, dev_grads, weight=1 - momentum)
            else:
                torch._foreach_mul_(dev_exp_avgs, scalar=momentum)
                torch._foreach_add_(dev_exp_avgs, dev_grads, alpha=1)

            if kahan_sum and dtype in [torch.float16, torch.bfloat16]:
                # SGD with Momentum step
                torch._foreach_add_(dev_kahan_comps, dev_exp_avgs, alpha=-lr)

                # update weights with kahan compensation using dev_grads as temp buffer
                torch._foreach_copy_(dev_grads, dev_params)
                torch._foreach_add_(dev_params, dev_kahan_comps, alpha=1)

                # save error back to kahan compensation for next iteration
                torch._foreach_sub_(dev_grads, dev_params, alpha=1)
                torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=1)
            else:
                # SGD with Momentum step
                torch._foreach_add_(dev_params, dev_exp_avgs, alpha=-lr)
        else:
            if kahan_sum and dtype in [torch.float16, torch.bfloat16]:
                # SGD step
                torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=-lr)

                # update weights with kahan compensation using dev_grads as temp buffer
                torch._foreach_copy_(dev_grads, dev_params)
                torch._foreach_add_(dev_params, dev_kahan_comps, alpha=1)

                # save error back to kahan compensation for next iteration
                torch._foreach_sub_(dev_grads, dev_params, alpha=1)
                torch._foreach_add_(dev_kahan_comps, dev_grads, alpha=1)
            else:
                # SGD step
                torch._foreach_add_(dev_params, dev_grads, alpha=-lr)
