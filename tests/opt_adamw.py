"""AdamW optimizer definitions using the new OptTest/variants flow."""

from dataclasses import dataclass

import optimi
import torch
from tests import reference

from .config import BaseParams, NormalSpec, OptTest, OptTestType, TestSpec, Tolerance


@dataclass
class AdamWParams(BaseParams):
    betas: tuple[float, float] = (0.9, 0.99)
    eps: float = 1e-6


# Provide BASE with fully_decoupled_reference so decoupled_lr uses DecoupledAdamW
BASE = OptTest(
    name="adamw",
    optimi_class=optimi.AdamW,
    optimi_params=AdamWParams(),
    reference_class=torch.optim.AdamW,
    reference_params=AdamWParams(),
    fully_decoupled_reference=reference.DecoupledAdamW,
)

TESTS = [
    OptTest(
        name="adamw_quantize",
        optimi_class=optimi.AdamW,
        optimi_params=AdamWParams(quantize=True),
        reference_class=torch.optim.AdamW,
        reference_params=AdamWParams(),
        skip_tests=[OptTestType.gradient_release, OptTestType.accumulation],
        triton_only=True,
        only_dtypes=[torch.bfloat16],
        spec=TestSpec(
            normal=NormalSpec(
                iterations_gpu=8,
                tolerance={
                    torch.bfloat16: Tolerance(atol=2e-2, rtol=2e-2, max_error_rate=0.08),
                },
            )
        ),
    ),
]
