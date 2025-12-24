"""Lion optimizer tests in new OptTest format (manual list to match prior values)."""

from dataclasses import dataclass

import optimi
from tests.reference import lion as reference_lion

from .cases import BaseParams, OptTest


@dataclass
class LionParams(BaseParams):
    betas: tuple[float, float] = (0.9, 0.99)


TESTS = [
    OptTest(
        name="lion_base",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0),
    ),
    OptTest(
        name="lion_decoupled_wd",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1, decouple_wd=True),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1),
    ),
    OptTest(
        name="lion_decoupled_lr",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-5, decouple_lr=True),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1),
    ),
]
