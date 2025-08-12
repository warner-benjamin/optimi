"""RAdam optimizer test definitions."""

from dataclasses import dataclass

import optimi
import torch

from .framework import BaseParams, OptimizerTest


@dataclass
class RAdamParams(BaseParams):
    """Type-safe RAdam optimizer parameters."""

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


BASE_TEST = OptimizerTest(
    name="radam_base",
    optimi_class=optimi.RAdam,
    optimi_params=RAdamParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0),
    reference_class=torch.optim.RAdam,
    reference_params=RAdamParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0),
)
