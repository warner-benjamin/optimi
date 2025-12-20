"""Ranger optimizer tests using new Case format (base only)."""

from dataclasses import dataclass

import optimi
from tests import reference

from .cases import BaseParams, Case, TestType


@dataclass
class RangerParams(BaseParams):
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    k: int = 6  # Lookahead steps
    alpha: float = 0.5  # Lookahead alpha


TESTS = [
    Case(
        name="ranger_base",
        optimi_class=optimi.Ranger,
        optimi_params=RangerParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0),
        reference_class=reference.Ranger,
        reference_params=RangerParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0),
        # Match legacy longer gradient-release coverage due to Lookahead cadence.
        custom_iterations={TestType.gradient_release: 160},
    )
]
