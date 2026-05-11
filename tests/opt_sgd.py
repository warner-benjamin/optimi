"""SGD optimizer definitions using the new OptTest/variants flow (manual list)."""

from dataclasses import dataclass
from typing import Any

import optimi
import torch

from tests import reference

from .config import BaseParams, NormalSpec, OptTest, OptTestType, TestSpec, Tolerance


@dataclass
class SGDParams(BaseParams):
    momentum: float = 0.0
    dampening: bool = False  # Optimi uses bool instead of float
    torch_init: bool = False

    def to_reference_kwargs(self, reference_class: type) -> dict[str, Any]:
        kwargs = super().to_reference_kwargs(reference_class)
        # Convert dampening bool to float for reference optimizer
        if "dampening" in kwargs and isinstance(kwargs["dampening"], bool):
            kwargs["dampening"] = 0.9 if kwargs["dampening"] else 0.0
        return kwargs


# Manual list to mirror the original explicit coverage for SGD
TESTS = [
    OptTest(
        name="sgd_base",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(),
        skip_tests=[OptTestType.accumulation],
    ),
    OptTest(
        name="sgd_momentum",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(momentum=0.9),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(momentum=0.9),
    ),
    OptTest(
        name="sgd_dampening",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(momentum=0.9, dampening=True, torch_init=True),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(momentum=0.9, dampening=0.9),
    ),
    OptTest(
        name="sgd_weight_decay",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(momentum=0.9, weight_decay=1e-2),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(momentum=0.9, weight_decay=1e-2),
        skip_tests=[OptTestType.accumulation],
    ),
    OptTest(
        name="sgd_decoupled_lr",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(momentum=0.9, dampening=True, decouple_lr=True, weight_decay=1e-5, torch_init=True),
        reference_class=reference.DecoupledSGDW,
        reference_params=SGDParams(momentum=0.9, dampening=0.9, weight_decay=1e-5),
    ),
    OptTest(
        name="sgd_flash",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(flash=True),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(),
        skip_tests=[OptTestType.gradient_release, OptTestType.accumulation],
        any_precision=True,
        triton_only=True,
        only_dtypes=[torch.bfloat16],
        spec=TestSpec(
            normal=NormalSpec(
                iterations_gpu=8,
                tolerance={
                    torch.float32: Tolerance(atol=1e-6, rtol=1e-5, max_error_rate=5e-4),
                    torch.bfloat16: Tolerance(atol=2e-2, rtol=2e-2, max_error_rate=0.08),
                    torch.float16: Tolerance(atol=1e-3, rtol=1e-2, max_error_rate=0.08),
                },
            )
        ),
    ),
    OptTest(
        name="sgd_flash_momentum",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(momentum=0.9, flash=True),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(momentum=0.9),
        skip_tests=[OptTestType.gradient_release, OptTestType.accumulation],
        any_precision=True,
        triton_only=True,
        only_dtypes=[torch.bfloat16],
        spec=TestSpec(
            normal=NormalSpec(
                iterations_gpu=8,
                tolerance={
                    torch.float32: Tolerance(atol=1e-6, rtol=1e-5, max_error_rate=5e-4),
                    torch.bfloat16: Tolerance(atol=2e-2, rtol=2e-2, max_error_rate=0.08),
                    torch.float16: Tolerance(atol=1e-3, rtol=1e-2, max_error_rate=0.08),
                },
            )
        ),
    ),
    OptTest(
        name="sgd_quantize_momentum",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(momentum=0.9, quantize=True),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(momentum=0.9),
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
    OptTest(
        name="sgd_stochastic_momentum",
        optimi_class=optimi.SGD,
        optimi_params=SGDParams(momentum=0.9, flash=True, stochastic_rounding=True),
        reference_class=torch.optim.SGD,
        reference_params=SGDParams(momentum=0.9),
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
