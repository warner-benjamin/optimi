"""Adan optimizer test definitions."""

from dataclasses import dataclass
from typing import Any

import optimi
from tests import reference

from .framework import BaseParams, OptimizerTest


@dataclass
class AdanParams(BaseParams):
    """Type-safe Adan optimizer parameters."""

    betas: tuple[float, float, float] = (0.98, 0.92, 0.99)
    eps: float = 1e-8
    weight_decouple: bool = False  # For adam_wd variant (maps to no_prox in reference)
    adam_wd: bool = False  # For optimi optimizer

    def to_reference_kwargs(self, reference_class: type) -> dict[str, Any]:
        """Adan needs special parameter conversion for no_prox."""
        kwargs = super().to_reference_kwargs(reference_class)

        # Convert weight_decouple to no_prox for reference optimizer
        if "weight_decouple" in kwargs:
            kwargs["no_prox"] = kwargs.pop("weight_decouple")

        # Remove adam_wd as it's not used by reference
        kwargs.pop("adam_wd", None)

        return kwargs


# Define all Adan test variants explicitly to match original tests
ALL_TESTS = [
    OptimizerTest(
        name="adan_base",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-6, weight_decay=0),
        reference_class=reference.Adan,
        reference_params=AdanParams(lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-6),
        custom_iterations={"correctness": 20},  # Adan bfloat16 updates are noisier, use fewer iterations for GPU
    ),
    OptimizerTest(
        name="adan_weight_decay",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-6, weight_decay=2e-2),
        reference_class=reference.Adan,
        reference_params=AdanParams(lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-6, weight_decay=2e-2),
        custom_iterations={"correctness": 20},
    ),
    OptimizerTest(
        name="adan_adam_wd",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-6, weight_decay=2e-2, adam_wd=True),
        reference_class=reference.Adan,
        reference_params=AdanParams(
            lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-6, weight_decay=2e-2, weight_decouple=True
        ),  # no_prox=True in reference
        custom_iterations={"correctness": 20},
    ),
    OptimizerTest(
        name="adan_decoupled_lr",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-6, weight_decay=2e-5, decouple_lr=True),
        reference_class=reference.Adan,
        reference_params=AdanParams(lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-6, weight_decay=2e-2),
        custom_iterations={"correctness": 20},
    ),
]

# Set BASE_TEST for auto-generation compatibility
BASE_TEST = ALL_TESTS[0]
