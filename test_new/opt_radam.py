"""RAdam optimizer definitions using the new Case/variants flow."""

import inspect
from dataclasses import dataclass, field

import optimi
import torch

from .cases import BaseParams, Case, Tolerance


@dataclass
class RAdamParams(BaseParams):
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    decoupled_weight_decay: bool = field(default=False)

    def __post_init__(self):
        if self.decouple_wd or self.decouple_lr:
            self.decoupled_weight_decay = True


BASE = Case(
    name="radam",
    optimi_class=optimi.RAdam,
    optimi_params=RAdamParams(lr=1e-3, betas=(0.9, 0.99), weight_decay=0),
    reference_class=torch.optim.RAdam,
    reference_params=RAdamParams(lr=1e-3, betas=(0.9, 0.99), weight_decay=0),
    custom_tolerances={torch.float32: Tolerance(max_error_rate=0.001)},
    test_decoupled_wd="decoupled_weight_decay" in inspect.signature(torch.optim.RAdam.__init__).parameters,
)
