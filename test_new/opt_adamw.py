"""AdamW optimizer definitions using the new Case/variants flow."""

from dataclasses import dataclass

import optimi
import torch
from tests import reference

from .cases import BaseParams, Case


@dataclass
class AdamWParams(BaseParams):
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-6


# Provide BASE with fully_decoupled_reference so decoupled_lr uses DecoupledAdamW
BASE = Case(
    name="adamw",
    optimi_class=optimi.AdamW,
    optimi_params=AdamWParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0),
    reference_class=torch.optim.AdamW,
    reference_params=AdamWParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0),
    fully_decoupled_reference=reference.DecoupledAdamW,
)
