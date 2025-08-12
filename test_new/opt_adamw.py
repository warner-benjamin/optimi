"""AdamW optimizer test definitions."""

from dataclasses import dataclass

import optimi
import torch
from tests import reference

from .framework import BaseParams, OptimizerTest


@dataclass
class AdamWParams(BaseParams):
    """Type-safe AdamW optimizer parameters."""

    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8


BASE_TEST = OptimizerTest(
    name="adamw_base",
    optimi_class=optimi.AdamW,
    optimi_params=AdamWParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0),
    reference_class=torch.optim.AdamW,
    fully_decoupled_reference=reference.DecoupledAdamW,
)
