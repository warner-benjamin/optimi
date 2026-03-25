"""Adan optimizer tests using the new OptTest format (manual list)."""

from dataclasses import dataclass
from typing import Any

import optimi
import torch
from tests import reference

from .config import BaseParams, DeviceType, OptTest, OptTestType, Tolerance, with_updated_spec


@dataclass
class AdanParams(BaseParams):
    betas: tuple[float, float, float] = (0.98, 0.92, 0.99)
    eps: float = 1e-5
    weight_decouple: bool = False  # For adam_wd variant (maps to no_prox in reference)
    adam_wd: bool = False  # For optimi optimizer

    def to_reference_kwargs(self, reference_class: type) -> dict[str, Any]:
        kwargs = super().to_reference_kwargs(reference_class)
        if "weight_decouple" in kwargs:
            kwargs["no_prox"] = kwargs.pop("weight_decouple")
        kwargs.pop("adam_wd", None)
        return kwargs


_custom_iterations = {(OptTestType.normal, DeviceType.gpu, torch.bfloat16): 20}
_spec = with_updated_spec(
    spec=None,
    test_type=OptTestType.normal,
    tolerances_override={torch.float16: Tolerance(atol=1.4e-4, rtol=1.4e-3, max_error_rate=0.01)},
)
TESTS = [
    OptTest(
        name="adan_base",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(),
        reference_class=reference.Adan,
        reference_params=AdanParams(),
        custom_iterations=_custom_iterations,
        spec=_spec,
    ),
    OptTest(
        name="adan_weight_decay",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(weight_decay=2e-2),
        reference_class=reference.Adan,
        reference_params=AdanParams(weight_decay=2e-2),
        custom_iterations=_custom_iterations,
        spec=_spec,
    ),
    OptTest(
        name="adan_adam_wd",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(weight_decay=2e-2, adam_wd=True),
        reference_class=reference.Adan,
        reference_params=AdanParams(weight_decay=2e-2, weight_decouple=True),
        custom_iterations=_custom_iterations,
        spec=_spec,
    ),
    OptTest(
        name="adan_decoupled_lr",
        optimi_class=optimi.Adan,
        optimi_params=AdanParams(weight_decay=2e-5, decouple_lr=True),
        reference_class=reference.Adan,
        reference_params=AdanParams(weight_decay=2e-2),
        custom_iterations=_custom_iterations,
        spec=_spec,
    ),
]
