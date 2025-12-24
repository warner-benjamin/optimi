"""StableAdamW optimizer definitions using new OptTest/variants flow."""

from dataclasses import dataclass

import optimi
from tests import reference

from .cases import BaseParams, OptTest


@dataclass
class StableAdamWParams(BaseParams):
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8


BASE = OptTest(
    name="stableadamw",
    optimi_class=optimi.StableAdamW,
    optimi_params=StableAdamWParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0),
    reference_class=reference.StableAdamWUnfused,
    reference_params=StableAdamWParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0),
)
