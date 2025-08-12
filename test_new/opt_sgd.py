"""SGD optimizer test definitions with custom parameter handling."""

from dataclasses import dataclass
from typing import Any

import optimi
import torch
from tests import reference

from .framework import BaseParams, OptimizerTest


@dataclass
class SGDParams(BaseParams):
    """Type-safe SGD optimizer parameters."""

    momentum: float = 0.0
    dampening: bool = False  # Optimi uses bool instead of float
    torch_init: bool = False

    def to_reference_kwargs(self, reference_class: type) -> dict[str, Any]:
        """SGD needs special dampening conversion for reference optimizer."""
        kwargs = super().to_reference_kwargs(reference_class)

        # Convert dampening bool to float for reference optimizer
        if "dampening" in kwargs and isinstance(kwargs["dampening"], bool):
            kwargs["dampening"] = 0.9 if kwargs["dampening"] else 0.0

        return kwargs


# Define all SGD test variants explicitly
ALL_TESTS = [
    OptimizerTest(
        name="sgd_base",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(lr=1e-3, momentum=0, dampening=False, weight_decay=0),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(lr=1e-3, momentum=0, dampening=0, weight_decay=0),
        skip_tests=["accumulation"],  # SGD base skips accumulation tests
    ),
    OptimizerTest(
        name="sgd_momentum",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(lr=1e-3, momentum=0.9, dampening=False, weight_decay=0),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(lr=1e-3, momentum=0.9, dampening=0, weight_decay=0),
    ),
    OptimizerTest(
        name="sgd_dampening",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(lr=1e-3, momentum=0.9, dampening=True, weight_decay=0, torch_init=True),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(lr=1e-3, momentum=0.9, dampening=0.9, weight_decay=0),
    ),
    OptimizerTest(
        name="sgd_weight_decay",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(lr=1e-3, momentum=0.9, dampening=False, weight_decay=1e-2, decouple_wd=False),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(lr=1e-3, momentum=0.9, dampening=0, weight_decay=1e-2),
        skip_tests=["accumulation"],  # SGD with L2 weight decay skips accumulation tests
    ),
    OptimizerTest(
        name="sgd_decoupled_lr",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(lr=1e-3, momentum=0.9, dampening=True, decouple_lr=True, weight_decay=1e-5, torch_init=True),
        reference_class=reference.DecoupledSGDW,
        reference_params=SGDParams(lr=1e-3, momentum=0.9, dampening=0.9, weight_decay=1e-5),
        custom_iterations={"accumulation": 20},  # SGD uses fewer iterations for accumulation
    ),
]

# Set BASE_TEST for auto-generation compatibility
BASE_TEST = ALL_TESTS[0]
