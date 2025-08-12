"""Lion optimizer test definitions."""

from dataclasses import dataclass

import optimi
from tests.reference import lion as reference_lion

from .framework import BaseParams, OptimizerTest


@dataclass
class LionParams(BaseParams):
    """Type-safe Lion optimizer parameters."""

    betas: tuple[float, float] = (0.9, 0.99)


BASE_TEST = OptimizerTest(
    name="lion_base",
    optimi_class=optimi.Lion,
    optimi_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0),
    reference_class=reference_lion.Lion,
    reference_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0),
)


# Define all Adan test variants explicitly to match original tests
ALL_TESTS = [
    OptimizerTest(
        name="lion_base",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0),
    ),
    OptimizerTest(
        name="lion_decoupled_wd",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1, decouple_wd=True),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1),
    ),
    OptimizerTest(
        name="lion_decoupled_lr",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-5, decouple_lr=True),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1),
    ),
]
