"""RAdam optimizer definitions using the new OptTest/variants flow."""

import inspect
from dataclasses import dataclass, field

import optimi
import torch

from .config import BaseParams, OptTest, OptTestType, Tolerance, with_updated_spec


@dataclass
class RAdamParams(BaseParams):
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-8
    decoupled_weight_decay: bool = field(default=False)

    def __post_init__(self):
        if self.decouple_wd or self.decouple_lr:
            self.decoupled_weight_decay = True


BASE = OptTest(
    name="radam",
    optimi_class=optimi.RAdam,
    optimi_params=RAdamParams(),
    reference_class=torch.optim.RAdam,
    reference_params=RAdamParams(),
    spec=with_updated_spec(spec=None, test_type=OptTestType.normal, tolerances_override={torch.float32: Tolerance(max_error_rate=0.001)}),
    test_decoupled_wd="decoupled_weight_decay" in inspect.signature(torch.optim.RAdam.__init__).parameters,
)
