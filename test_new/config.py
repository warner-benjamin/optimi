from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .cases import Tolerance


@dataclass(frozen=True)
class CorrectnessDefaults:
    cpu_iterations: int = 20
    gpu_iterations: int = 40
    # Special-case: Adan in bf16 on GPU is noisier; align to 20
    adan_bf16_gpu_iterations: int = 20

    cpu_dims: tuple[int, int] = (64, 128)
    gpu_dims: tuple[int, int] = (256, 512)

    cpu_batch_size: int = 1
    gpu_batch_size: int = 32

    cpu_max_error_count: int = 2
    gpu_max_error_count: int = 5


@dataclass(frozen=True)
class GradientReleaseDefaults:
    iterations: int = 40
    dims: tuple[int, int] = (128, 256)
    batch_size: int = 32
    max_error_count: int = 12  # more lenient for noisy updates

    baseline_tolerance: dict[torch.dtype, Tolerance] = field(
        default_factory=lambda: {
            torch.float32: Tolerance(atol=1e-6, rtol=1e-5, max_error_rate=5e-4),
            torch.bfloat16: Tolerance(atol=1e-3, rtol=1e-2, max_error_rate=0.01),
            torch.float16: Tolerance(atol=1e-4, rtol=1e-3, max_error_rate=0.01),
        }
    )


@dataclass(frozen=True)
class AccumulationDefaults:
    iterations: int = 40
    dims: tuple[int, int] = (128, 256)
    batch_size: int = 32
    tolerance: Tolerance = field(default_factory=lambda: Tolerance(rtol=1e-2, atol=1e-2))
    max_error_rate: float = 0.035
    gradient_accumulation_steps: int = 4


@dataclass(frozen=True)
class TestDefaults:
    correctness: CorrectnessDefaults = CorrectnessDefaults()
    gradient_release: GradientReleaseDefaults = GradientReleaseDefaults()
    accumulation: AccumulationDefaults = AccumulationDefaults()


# Single place to tweak numbers used by runners
DEFAULTS = TestDefaults()
