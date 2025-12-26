"""Ranger optimizer tests using new OptTest format (base only)."""

from dataclasses import dataclass

import optimi
from tests import reference

from .config import BaseParams, OptTest, OptTestType


@dataclass
class RangerParams(BaseParams):
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    k: int = 6  # Lookahead steps
    alpha: float = 0.5  # Lookahead alpha


TESTS = [
    OptTest(
        name="ranger_base",
        optimi_class=optimi.Ranger,
        optimi_params=RangerParams(),
        reference_class=reference.Ranger,
        reference_params=RangerParams(),
        # Match legacy longer gradient-release coverage due to Lookahead cadence.
        custom_iterations={OptTestType.gradient_release: 160},
    )
]
