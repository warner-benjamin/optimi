"""Lion optimizer tests in new OptTest format (manual list to match prior values)."""

from dataclasses import dataclass

import optimi
from tests.reference import lion as reference_lion

from .config import BaseParams, OptTest


@dataclass
class LionParams(BaseParams):
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.99)


TESTS = [
    OptTest(
        name="lion_base",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(),
    ),
    OptTest(
        name="lion_decoupled_wd",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(weight_decay=0.1, decouple_wd=True),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(weight_decay=0.1),
    ),
    OptTest(
        name="lion_decoupled_lr",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(weight_decay=1e-5, decouple_lr=True),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(weight_decay=0.1),
    ),
]
