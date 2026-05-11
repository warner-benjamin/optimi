"""Lion optimizer tests in new OptTest format (manual list to match prior values)."""

from dataclasses import dataclass

import optimi
import torch

from tests.reference import lion as reference_lion

from .config import BaseParams, NormalSpec, OptTest, OptTestType, TestSpec, Tolerance


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
    OptTest(
        name="lion_flash",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(flash=True),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(),
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
        name="lion_quantize",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(quantize=True),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(),
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
        name="lion_stochastic",
        optimi_class=optimi.Lion,
        optimi_params=LionParams(flash=True, stochastic_rounding=True),
        reference_class=reference_lion.Lion,
        reference_params=LionParams(),
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
