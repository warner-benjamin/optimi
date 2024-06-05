# Optimizer utilities

from __future__ import annotations

from typing import Any, Iterable

import torch
from packaging.version import parse
from torch import nn

MIN_TORCH_2_1 = parse(torch.__version__) >= parse("2.1")


def debias(beta: float, step: int) -> float:
    """Adam-style debias correction. Returns `1 - beta ** step`."""
    return 1 - beta**step


def debias_beta(beta: float, step: int) -> float:
    """Applies the Adam-style debias correction into beta.

    Simplified version of `betahat = beta*(1-beta**(step-1))/(1-beta**step)`
    """
    return (beta**step - beta) / (beta**step - 1)


# modified from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
# Copyright 2019 Ross Wightman, Apache-2.0 License
def param_groups_weight_decay(
    model: nn.Module, weight_decay: float = 1e-2, additional_layers: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """Creates parameter groups, excluding bias and normalization layers from weight decay.

    Parameters:
        model: Model to optimize
        weight_decay: Weight decay coefficient (default: 1e-2)
        additional_layers: Additional layer names to exclude from weight decay (default: None)

    Returns:
        List of parameter groups with and without weight decay.
    """
    additional_layers = set(additional_layers) if additional_layers is not None else set()
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in additional_layers:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
