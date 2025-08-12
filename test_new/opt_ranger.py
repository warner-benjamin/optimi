"""Ranger optimizer test definitions."""

from dataclasses import dataclass

import optimi
from tests import reference

from .framework import BaseParams, OptimizerTest


@dataclass
class RangerParams(BaseParams):
    """Type-safe Ranger optimizer parameters."""

    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    k: int = 6  # Lookahead steps
    alpha: float = 0.5  # Lookahead alpha


# Ranger only has base test - reference doesn't perform normal weight decay step
ALL_TESTS = [
    OptimizerTest(
        name="ranger_base",
        optimi_class=optimi.Ranger,
        optimi_params=RangerParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0),
        reference_class=reference.Ranger,
        reference_params=RangerParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0),
        custom_iterations={"gradient_release": 160},  # Ranger needs longer testing due to lookahead step
    )
]

# Set BASE_TEST for auto-generation compatibility
BASE_TEST = ALL_TESTS[0]
