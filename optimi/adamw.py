# Copyright (c) 2023-present Benjamin Warner
# SPDX-License-Identifier: MIT

# Based on PyTorch Optimizers
# PyTorch - PyTorch BSD-style license - Copyright (c) 2013-present PyTorch contributors

# Kahan summation inspired by Torch Distributed Experimental's `AnyPrecisionAdamW`
# torchdistX - BSD 3-Clause License - Copyright (c) Meta Platforms, Inc. and affiliates

# Learning rate decoupled weight decay inspired by Composer's `DecoupledSGDW` & `DecoupledAdamW`
# Composer - Apache License 2.0 - Copyright (c) 2022 MosaicML Composer authors

from collections.abc import Iterable

from torch import Tensor

from optimi import Adam, adam
from optimi.flash import FlashPrecision

__all__ = ["AdamW", "adamw"]


class AdamW(Adam):
    """AdamW optimizer: Adam with decoupled weight decay.

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
        weight_decay: float = 1e-2,
        eps: float = 1e-6,
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
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            decouple_wd=True,
            decouple_lr=decouple_lr,
            max_lr=max_lr,
            kahan_sum=kahan_sum,
            flash=flash,
            quantize=quantize,
            stochastic_rounding=stochastic_rounding,
            foreach=foreach,
            triton=triton,
            gradient_release=gradient_release,
        )


def adamw(
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
    """Functional API to apply an AdamW optimization step.

    See `optimi.AdamW` for more details.

    Args:
        params: Parameters to update
        grads: Parameter gradients
        exp_avgs: Gradient moving averages
        exp_avg_sqs: Squared gradient moving averages
        kahan_comps: Kahan summation compensations
        flash_comps: FlashOptim parameter compensations
        lr: Learning rate
        beta1: Gradient moving average factor
        beta2: Squared gradient moving average factor
        weight_decay: Weight decay coefficient
        eps: Added to denominator to improve numerical stability
        step: Step counter used for bias correction
        decouple_lr: Apply fully decoupled weight decay
        max_lr: Maximum scheduled learning rate for `decouple_lr`
        kahan_sum: Enables Kahan summation for low precision `params`
        quantize: Reserved for optimizer-state quantization
        stochastic_rounding: Reserved for FlashOptim stochastic rounding
        foreach: Removed; True raises ValueError
        triton: Enables the faster Triton implementation
        gradient_release: Fuses optimizer step as part of the parameter's backward pass
        optimizer_accumulation: Accumulate gradients into state during gradient release step
    """
    adam(
        params=params,
        grads=grads,
        exp_avgs=exp_avgs,
        exp_avg_sqs=exp_avg_sqs,
        kahan_comps=kahan_comps,
        flash_comps=flash_comps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        step=step,
        decouple_wd=True,
        decouple_lr=decouple_lr,
        max_lr=max_lr,
        kahan_sum=kahan_sum,
        quantize=quantize,
        stochastic_rounding=stochastic_rounding,
        foreach=foreach,
        triton=triton,
        gradient_release=gradient_release,
        optimizer_accumulation=optimizer_accumulation,
    )
