"""RAdam optimizer test definitions."""

from dataclasses import dataclass, field
from typing import Any

import optimi
import torch

from .framework import BaseParams, OptimizerTest, ToleranceConfig


@dataclass
class RAdamParams(BaseParams):
    """Type-safe RAdam optimizer parameters."""

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-6
    decoupled_weight_decay: bool = field(default=False)

    def __post_init__(self):
        if self.decouple_wd:
            self.decoupled_weight_decay = True
        elif self.decouple_lr:
            self.decoupled_weight_decay = True


BASE_TEST = OptimizerTest(
    name="radam_base",
    optimi_class=optimi.RAdam,
    optimi_params=RAdamParams(lr=1e-3, betas=(0.9, 0.99), weight_decay=0),
    reference_class=torch.optim.RAdam,
    reference_params=RAdamParams(lr=1e-3, betas=(0.9, 0.99), weight_decay=0),
    custom_tolerances={torch.float32: ToleranceConfig(max_error_rate=0.001)},
)
