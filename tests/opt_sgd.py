"""SGD optimizer definitions using the new OptTest/variants flow (manual list)."""

from dataclasses import dataclass
from typing import Any

import optimi
import torch
from tests import reference

from .config import BaseParams, OptTest, OptTestType


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
    OptTest(
        name="sgd_base",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(),
        skip_tests=[OptTestType.accumulation],
    ),
    OptTest(
        name="sgd_momentum",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(momentum=0.9),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(momentum=0.9),
    ),
    OptTest(
        name="sgd_dampening",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(momentum=0.9, dampening=True, torch_init=True),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(momentum=0.9, dampening=0.9),
    ),
    OptTest(
        name="sgd_weight_decay",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(momentum=0.9, weight_decay=1e-2),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(momentum=0.9, weight_decay=1e-2),
        skip_tests=[OptTestType.accumulation],
    ),
    OptTest(
        name="sgd_decoupled_lr",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(momentum=0.9, dampening=True, decouple_lr=True, weight_decay=1e-5, torch_init=True),
        reference_class=reference.DecoupledSGDW,
        reference_params=SGDParams(momentum=0.9, dampening=0.9, weight_decay=1e-5),
    ),
]
