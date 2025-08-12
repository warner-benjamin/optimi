"""StableAdamW optimizer test definitions."""

from dataclasses import dataclass

import optimi
from tests import reference

from .framework import BaseParams, OptimizerTest


@dataclass
class StableAdamWParams(BaseParams):
    """Type-safe StableAdamW optimizer parameters."""

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


BASE_TEST = OptimizerTest(
    name="stableadamw_base",
    optimi_class=optimi.StableAdamW,
    optimi_params=StableAdamWParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0),
    reference_class=reference.StableAdamWUnfused,
    reference_params=StableAdamWParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0),
)
