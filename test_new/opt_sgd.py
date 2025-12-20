"""SGD optimizer definitions using the new Case/variants flow (manual list)."""

from dataclasses import dataclass
from typing import Any

import optimi
import torch
from tests import reference

from .cases import BaseParams, Case, TestType


@dataclass
class SGDParams(BaseParams):
    momentum: float = 0.0
    dampening: bool = False  # Optimi uses bool instead of float
    torch_init: bool = False

    def to_reference_kwargs(self, reference_class: type) -> dict[str, Any]:
        kwargs = super().to_reference_kwargs(reference_class)
        # Convert dampening bool to float for reference optimizer
        if "dampening" in kwargs and isinstance(kwargs["dampening"], bool):
            kwargs["dampening"] = 0.9 if kwargs["dampening"] else 0.0
        return kwargs


# Manual list to mirror the original explicit coverage for SGD
TESTS = [
    Case(
        name="sgd_base",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(lr=1e-3, momentum=0, dampening=False, weight_decay=0),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(lr=1e-3, momentum=0, dampening=0, weight_decay=0),
        skip_tests=[TestType.accumulation],
    ),
    Case(
        name="sgd_momentum",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(lr=1e-3, momentum=0.9, dampening=False, weight_decay=0),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(lr=1e-3, momentum=0.9, dampening=0, weight_decay=0),
    ),
    Case(
        name="sgd_dampening",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(lr=1e-3, momentum=0.9, dampening=True, weight_decay=0, torch_init=True),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(lr=1e-3, momentum=0.9, dampening=0.9, weight_decay=0),
    ),
    Case(
        name="sgd_weight_decay",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(lr=1e-3, momentum=0.9, dampening=False, weight_decay=1e-2, decouple_wd=False),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(lr=1e-3, momentum=0.9, dampening=0, weight_decay=1e-2),
        skip_tests=[TestType.accumulation],
    ),
    Case(
        name="sgd_decoupled_lr",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(lr=1e-3, momentum=0.9, dampening=True, decouple_lr=True, weight_decay=1e-5, torch_init=True),
        reference_class=reference.DecoupledSGDW,
        reference_params=SGDParams(lr=1e-3, momentum=0.9, dampening=0.9, weight_decay=1e-5),
    ),
]
