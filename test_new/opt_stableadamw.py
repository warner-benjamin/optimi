"""StableAdamW optimizer definitions using new OptTest/variants flow."""

from dataclasses import dataclass

import optimi
from tests import reference

from .config import BaseParams, OptTest


@dataclass
class StableAdamWParams(BaseParams):
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-6


BASE = OptTest(
    name="stableadamw",
    optimi_class=optimi.StableAdamW,
    optimi_params=StableAdamWParams(),
    reference_class=reference.StableAdamWUnfused,
    reference_params=StableAdamWParams(),
)
