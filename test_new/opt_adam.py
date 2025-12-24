"""Adam optimizer definitions using the new OptTest/variants flow."""

from dataclasses import dataclass

import optimi
import torch

from .cases import BaseParams, OptTest


@dataclass
class AdamParams(BaseParams):
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-6


# Provide BASE so the framework generates base/l2/decoupled variants as applicable.
# Disable decoupled WD/LR generation as this is tested in AdamW tests.
BASE = OptTest(
    name="adam",
    optimi_class=optimi.Adam,
    optimi_params=AdamParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.0),
    reference_class=torch.optim.Adam,
    reference_params=AdamParams(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.0),
    test_decoupled_wd=False,
)
