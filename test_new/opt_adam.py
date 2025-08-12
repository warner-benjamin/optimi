"""Adam optimizer test definitions."""

from dataclasses import dataclass

import optimi
import torch

from .framework import BaseParams, OptimizerTest


@dataclass
class AdamParams(BaseParams):
    """Type-safe Adam optimizer parameters."""

    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-6


BASE_TEST = OptimizerTest(
    name="adam_base",
    optimi_class=optimi.Adam,
    optimi_params=AdamParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0),
    reference_class=torch.optim.Adam,
    test_decoupled_wd=False,
)
