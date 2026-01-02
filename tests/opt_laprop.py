"""LaProp optimizer tests using the new OptTest format (manual list)."""

from dataclasses import dataclass

import optimi
from tests import reference

from .config import BaseParams, OptTest


@dataclass
class LaPropParams(BaseParams):
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-6


BASE = OptTest(
    name="laprop",
    optimi_class=optimi.LaProp,
    optimi_params=LaPropParams(),
    reference_class=reference.LaProp,
    reference_params=LaPropParams(),
)