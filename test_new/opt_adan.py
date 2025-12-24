"""Adan optimizer tests using the new OptTest format (manual list)."""

from dataclasses import dataclass
from typing import Any

import optimi
from tests import reference

from .config import BaseParams, OptTest


@dataclass
class AdanParams(BaseParams):
    betas: tuple[float, float, float] = (0.98, 0.92, 0.99)
    eps: float = 1e-6
    weight_decouple: bool = False  # For adam_wd variant (maps to no_prox in reference)
    adam_wd: bool = False  # For optimi optimizer

    def to_reference_kwargs(self, reference_class: type) -> dict[str, Any]:
        kwargs = super().to_reference_kwargs(reference_class)
        if "weight_decouple" in kwargs:
            kwargs["no_prox"] = kwargs.pop("weight_decouple")
        kwargs.pop("adam_wd", None)
        return kwargs


TESTS = [
    OptTest(
        name="adan_base",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(),
        reference_class=reference.Adan,
        reference_params=AdanParams(),
    ),
    OptTest(
        name="adan_weight_decay",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(weight_decay=2e-2),
        reference_class=reference.Adan,
        reference_params=AdanParams(weight_decay=2e-2),
    ),
    OptTest(
        name="adan_adam_wd",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(weight_decay=2e-2, adam_wd=True),
        reference_class=reference.Adan,
        reference_params=AdanParams(weight_decay=2e-2, weight_decouple=True),
    ),
    OptTest(
        name="adan_decoupled_lr",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(weight_decay=2e-5, decouple_lr=True),
        reference_class=reference.Adan,
        reference_params=AdanParams(weight_decay=2e-2),
    ),
]
