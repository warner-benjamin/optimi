"""AnyAdam optimizer test definitions for Kahan summation precision tests."""

from dataclasses import dataclass

import optimi
import torch
from tests.reference import AnyPrecisionAdamW

from .framework import BaseParams, OptimizerTest, ToleranceConfig


@dataclass
class AnyAdamParams(BaseParams):
    """Type-safe AnyAdam optimizer parameters with Kahan summation support."""

    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    kahan_sum: bool = False
    use_kahan_summation: bool = False

    def to_reference_kwargs(self, reference_class: type) -> dict:
        """Convert parameters for AnyPrecisionAdamW reference."""
        kwargs = super().to_reference_kwargs(reference_class)

        # AnyPrecisionAdamW uses use_kahan_summation instead of kahan_sum
        if "kahan_sum" in kwargs:
            kwargs["use_kahan_summation"] = kwargs.pop("kahan_sum")

        # Set default precision dtypes for AnyPrecisionAdamW
        if reference_class.__name__ == "AnyPrecisionAdamW":
            kwargs.setdefault("momentum_dtype", torch.bfloat16)
            kwargs.setdefault("variance_dtype", torch.bfloat16)
            kwargs.setdefault("compensation_buffer_dtype", torch.bfloat16)

        return kwargs


ALL_TESTS = [
    OptimizerTest(
        name="anyadam_kahan",
        optimi_class=optimi.Adam,
        optimi_params=AnyAdamParams(
            lr=1e-3,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=0,
            kahan_sum=True,
        ),
        reference_class=AnyPrecisionAdamW,
        reference_params=AnyAdamParams(
            lr=1e-3,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=0,
            use_kahan_summation=True,
        ),
        only_dtypes=[torch.bfloat16],
        any_precision=True,
        custom_tolerances={torch.bfloat16: ToleranceConfig(rtol=2e-2, atol=2e-3, equal_nan=False)},
    ),
    OptimizerTest(
        name="anyadam_kahan_wd",
        optimi_class=optimi.Adam,
        optimi_params=AnyAdamParams(
            lr=1e-3,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=0.01,
            kahan_sum=True,
        ),
        reference_class=AnyPrecisionAdamW,
        reference_params=AnyAdamParams(
            lr=1e-3,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=0.01,
            use_kahan_summation=True,
        ),
        only_dtypes=[torch.bfloat16],
        any_precision=True,
        custom_tolerances={torch.bfloat16: ToleranceConfig(rtol=5e-2, atol=1e-2, equal_nan=False)},
    ),
    OptimizerTest(
        name="anyadam_kahan_decoupled_lr",
        optimi_class=optimi.Adam,
        optimi_params=AnyAdamParams(
            lr=1e-3,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=1e-5,
            decouple_lr=True,
            kahan_sum=True,
        ),
        reference_class=AnyPrecisionAdamW,
        reference_params=AnyAdamParams(
            lr=1e-3,
            betas=(0.9, 0.99),
            eps=1e-6,
            weight_decay=1e-2,
            use_kahan_summation=True,
        ),
        only_dtypes=[torch.bfloat16],
        any_precision=True,
        custom_tolerances={torch.bfloat16: ToleranceConfig(rtol=2e-2, atol=2e-3, equal_nan=False)},
    ),
]

# For compatibility with auto-generation system
BASE_TEST = ALL_TESTS[0]
