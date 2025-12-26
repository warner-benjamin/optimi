"""AnyAdam optimizer tests using the new OptTest format (manual list)."""

from dataclasses import dataclass

import optimi
import torch
from tests.reference import AnyPrecisionAdamW

from .config import BaseParams, OptTest, OptTestType, Tolerance, with_updated_spec


@dataclass
class AnyAdamParams(BaseParams):
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-6
    kahan_sum: bool = False
    use_kahan_summation: bool = False

    def to_reference_kwargs(self, reference_class: type) -> dict:
        kwargs = super().to_reference_kwargs(reference_class)
        if "kahan_sum" in kwargs:
            kwargs["use_kahan_summation"] = kwargs.pop("kahan_sum")
        if reference_class.__name__ == "AnyPrecisionAdamW":
            kwargs.setdefault("momentum_dtype", torch.bfloat16)
            kwargs.setdefault("variance_dtype", torch.bfloat16)
            kwargs.setdefault("compensation_buffer_dtype", torch.bfloat16)
        return kwargs


TESTS = [
    OptTest(
        name="anyadam_kahan",
        optimi_class=optimi.Adam,
        optimi_params=AnyAdamParams(betas=(0.9, 0.99), kahan_sum=True),
        reference_class=AnyPrecisionAdamW,
        reference_params=AnyAdamParams(betas=(0.9, 0.99), use_kahan_summation=True),
        only_dtypes=[torch.bfloat16],
        any_precision=True,
        spec=with_updated_spec(
            spec=None,
            test_type=OptTestType.normal,
            tolerances_override={torch.bfloat16: Tolerance(rtol=2e-2, atol=2e-3, max_error_rate=0.01)},
        ),
    ),
    OptTest(
        name="anyadam_kahan_wd",
        optimi_class=optimi.AdamW,
        optimi_params=AnyAdamParams(betas=(0.9, 0.99), weight_decay=0.01, kahan_sum=True),
        reference_class=AnyPrecisionAdamW,
        reference_params=AnyAdamParams(betas=(0.9, 0.99), weight_decay=0.01, use_kahan_summation=True),
        only_dtypes=[torch.bfloat16],
        any_precision=True,
        spec=with_updated_spec(
            spec=None,
            test_type=OptTestType.normal,
            tolerances_override={torch.bfloat16: Tolerance(rtol=5e-2, atol=1e-2, max_error_rate=0.01)},
        ),
    ),
    OptTest(
        name="anyadam_kahan_decoupled_lr",
        optimi_class=optimi.AdamW,
        optimi_params=AnyAdamParams(betas=(0.9, 0.99), weight_decay=1e-5, decouple_lr=True, kahan_sum=True),
        reference_class=AnyPrecisionAdamW,
        reference_params=AnyAdamParams(betas=(0.9, 0.99), weight_decay=1e-2, use_kahan_summation=True),
        only_dtypes=[torch.bfloat16],
        any_precision=True,
        spec=with_updated_spec(
            spec=None,
            test_type=OptTestType.normal,
            tolerances_override={torch.bfloat16: Tolerance(rtol=2e-2, atol=2e-3, max_error_rate=0.01)},
        ),
    ),
]
