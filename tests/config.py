from __future__ import annotations

import importlib
import inspect
import warnings
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from optimi.optimizer import OptimiOptimizer
from torch.optim import Optimizer


from optimi.utils import MIN_TORCH_2_6


class OptTestType(Enum):
    normal = "normal"
    gradient_release = "gradient_release"
    accumulation = "accumulation"


class DeviceType(Enum):
    cpu = "cpu"
    gpu = "gpu"

    def is_available(self) -> bool:
        if self == DeviceType.cpu:
            return True
        if self == DeviceType.gpu:
            return (
                torch.cuda.is_available()
                or (hasattr(torch, "xpu") and torch.xpu.is_available())
                or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            )
        return False


class Backend(Enum):
    torch = "torch"
    triton = "triton"
    foreach = "foreach"

    def is_supported(self, device: DeviceType) -> bool:
        if self == Backend.triton:
            # Triton requires torch >= 2.6
            if not MIN_TORCH_2_6:
                return False
            # Triton not supported on CPU
            if device == DeviceType.cpu:
                return False
            # Triton not supported on MPS
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return False
            # Triton requires GPU/XPU
            if not (torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())):
                return False
        if self == Backend.foreach:
            # skip foreach on CPU
            if device == DeviceType.cpu:
                return False
            # forach has limited support on MPS/XPU
            if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) or (
                hasattr(torch, "xpu") and torch.xpu.is_available()
            ):
                return False
            # foreach wors best on CUDA devices
            if not torch.cuda.is_available():
                return False
        return True


@dataclass
class Tolerance:
    atol: float = 1e-6
    rtol: float = 1e-5
    max_error_rate: float = 5e-4
    equal_nan: bool = False


@dataclass()
class NormalSpec:
    iterations_cpu: int = 20
    iterations_gpu: int = 40
    batch_cpu: int = 1
    batch_gpu: int = 32
    max_error_cpu: int = 2
    max_error_gpu: int = 5

    tolerance: dict[torch.dtype, Tolerance] = field(
        default_factory=lambda: {
            torch.float32: Tolerance(atol=1e-6, rtol=1e-5, max_error_rate=5e-4),
            torch.bfloat16: Tolerance(atol=1e-3, rtol=1e-2, max_error_rate=0.01),
            torch.float16: Tolerance(atol=1e-4, rtol=1e-3, max_error_rate=0.01),
        }
    )


@dataclass()
class GradientReleaseSpec:
    iterations: int = 40
    batch: int = 32
    max_error_count: int = 12  # more lenient for noisy updates

    tolerance: dict[torch.dtype, Tolerance] = field(
        default_factory=lambda: {
            torch.float32: Tolerance(atol=1e-6, rtol=1e-5, max_error_rate=5e-4),
            torch.bfloat16: Tolerance(atol=1e-3, rtol=1e-2, max_error_rate=0.01),
            torch.float16: Tolerance(atol=1e-4, rtol=1e-3, max_error_rate=0.01),
        }
    )


@dataclass()
class AccumulationSpec:
    iterations: int = 40
    batch: int = 32
    max_error_rate: float = 0.035
    gradient_accumulation_steps: int = 4

    tolerance: dict[torch.dtype, Tolerance] = field(
        default_factory=lambda: {
            torch.float32: Tolerance(rtol=1e-2, atol=1e-2),
            torch.bfloat16: Tolerance(rtol=1e-2, atol=1e-2),
            torch.float16: Tolerance(rtol=1e-2, atol=1e-2),
        }
    )


@dataclass()
class TestSpec:
    normal: NormalSpec = field(default_factory=NormalSpec)
    gradient_release: GradientReleaseSpec = field(default_factory=GradientReleaseSpec)
    accumulation: AccumulationSpec = field(default_factory=AccumulationSpec)


def with_updated_spec(
    spec: TestSpec | NormalSpec | GradientReleaseSpec | AccumulationSpec | None,
    test_type: OptTestType | None = None,
    tolerances_override: dict[torch.dtype, Tolerance] | None = None,
) -> TestSpec:
    if isinstance(spec, (NormalSpec, GradientReleaseSpec, AccumulationSpec)):
        if isinstance(spec, NormalSpec):
            base = TestSpec(normal=spec)
        elif isinstance(spec, GradientReleaseSpec):
            base = TestSpec(gradient_release=spec)
        else:
            base = TestSpec(accumulation=spec)
    else:
        base = spec or TestSpec()

    if tolerances_override is None:
        tolerances_override = {}

    if test_type is None:
        return base

    if test_type == OptTestType.normal:
        merged = {**base.normal.tolerance, **tolerances_override}
        return replace(base, normal=replace(base.normal, tolerance=merged))
    if test_type == OptTestType.gradient_release:
        merged = {**base.gradient_release.tolerance, **tolerances_override}
        return replace(base, gradient_release=replace(base.gradient_release, tolerance=merged))
    if test_type == OptTestType.accumulation:
        merged = {**base.accumulation.tolerance, **tolerances_override}
        return replace(base, accumulation=replace(base.accumulation, tolerance=merged))
    raise ValueError(f"Unknown test type: {test_type}")


@dataclass
class BaseParams:
    lr: float = 1e-3
    weight_decay: float = 0.0
    decouple_wd: bool = False
    decouple_lr: bool = False
    triton: bool = False

    def with_(self, **overrides: Any) -> "BaseParams":
        return replace(self, **overrides)

    def _kwargs_for(self, cls: type | None) -> dict[str, Any]:
        if cls is None:
            return {}
        sig = inspect.signature(cls.__init__)
        ok = set(sig.parameters) - {"self"}
        values = asdict(self)
        if values.get("triton") and "triton" not in ok:
            warnings.warn(f"{cls.__name__} does not accept triton; ignoring BaseParams.triton=True.", RuntimeWarning)
        return {k: v for k, v in values.items() if k in ok}

    def to_optimi_kwargs(self, cls: type[OptimiOptimizer]) -> dict[str, Any]:
        return self._kwargs_for(cls)

    def to_reference_kwargs(self, cls: type[Optimizer]) -> dict[str, Any]:
        return self._kwargs_for(cls)


@dataclass
class OptTest:
    # Identification
    name: str  # e.g. "adam_base"

    # Classes + params
    optimi_class: type[OptimiOptimizer]
    optimi_params: BaseParams
    reference_class: type[Optimizer]
    reference_params: BaseParams | None = None

    # Optional fully decoupled reference for decoupled-lr variant
    fully_decoupled_reference: type[Optimizer] | None = None

    # Behavior / constraints
    skip_tests: list[OptTestType] = field(default_factory=list)
    any_precision: bool = False
    test_decoupled_wd: bool = True
    custom_iterations: dict[OptTestType | tuple[OptTestType, DeviceType] | tuple[OptTestType, DeviceType, torch.dtype], int] | None = None
    spec: TestSpec = field(default_factory=TestSpec)
    only_dtypes: list[torch.dtype] | None = None

    def __post_init__(self):
        if self.reference_params is None:
            self.reference_params = self.optimi_params

    @property
    def optimizer_name(self) -> str:
        return self.name.split("_", 1)[0]

    @property
    def variant_name(self) -> str:
        return self.name.split("_", 1)[1] if "_" in self.name else "base"

    def to_optimi_kwargs(self, backend: Backend | None = None) -> dict[str, Any]:
        kw = self.optimi_params.to_optimi_kwargs(self.optimi_class)

        # Centralize backend controls so runners don't mutate kwargs later
        if backend is not None:
            if backend == Backend.triton:
                kw["triton"] = True
                kw["foreach"] = False
            elif backend == Backend.torch:
                kw["triton"] = False
                kw["foreach"] = False
            elif backend == Backend.foreach:
                kw["triton"] = False
                kw["foreach"] = True
            else:
                raise ValueError(f"Unknown backend: {backend}")
        return kw

    def to_reference_kwargs(self, backend: Backend | None = None) -> dict[str, Any]:
        assert self.reference_params is not None
        kwargs = self.reference_params.to_reference_kwargs(self.reference_class)
        # Centralize fused handling for reference optimizers: when not testing
        # Optimi's Triton backend, avoid fused codepaths on the reference side
        # to mirror legacy parity expectations.
        if backend is not None and backend != Backend.triton:
            try:
                if "fused" in inspect.signature(self.reference_class.__init__).parameters:
                    kwargs = {**kwargs, "fused": False}
            except (ValueError, TypeError):
                pass
        return kwargs

    def supports_l2_weight_decay(self) -> bool:
        return "decouple_wd" in inspect.signature(self.optimi_class.__init__).parameters




def default_variants(base: OptTest) -> list[OptTest]:
    """Generate base + L2 + decoupled variants with minimal boilerplate."""
    out: list[OptTest] = []

    base_test = OptTest(
        name=f"{base.optimizer_name}_base",
        optimi_class=base.optimi_class,
        optimi_params=base.optimi_params.with_(weight_decay=0.0, decouple_wd=False, decouple_lr=False),
        reference_class=base.reference_class,
        reference_params=(base.reference_params or base.optimi_params).with_(weight_decay=0.0, decouple_wd=False, decouple_lr=False),
        test_decoupled_wd=base.test_decoupled_wd,
        skip_tests=list(base.skip_tests),
        any_precision=base.any_precision,
        custom_iterations=base.custom_iterations,
        spec=base.spec,
        only_dtypes=base.only_dtypes,
        fully_decoupled_reference=base.fully_decoupled_reference,
    )
    out.append(base_test)

    optimi_params = inspect.signature(base.optimi_class.__init__).parameters

    # L2 weight decay if optimizer supports decouple_wd arg
    if "decouple_wd" in optimi_params:
        out.append(
            replace(
                base_test,
                name=f"{base.optimizer_name}_l2_wd",
                optimi_params=base.optimi_params.with_(weight_decay=0.01, decouple_wd=False),
                reference_params=(base.reference_params or base.optimi_params).with_(weight_decay=0.01, decouple_wd=False),
            )
        )

    # Decoupled weight decay
    if base.test_decoupled_wd and "decouple_lr" in optimi_params:
        out.append(
            replace(
                base_test,
                name=f"{base.optimizer_name}_decoupled_wd",
                optimi_params=base.optimi_params.with_(weight_decay=0.01, decouple_wd=True),
                reference_params=(base.reference_params or base.optimi_params).with_(weight_decay=0.01, decouple_wd=True),
            )
        )

        # Decoupled LR (optionally swap reference class)
        ref_cls = base.fully_decoupled_reference or base.reference_class
        out.append(
            replace(
                base_test,
                name=f"{base.optimizer_name}_decoupled_lr",
                optimi_params=base.optimi_params.with_(weight_decay=1e-5, decouple_lr=True),
                reference_class=ref_cls,
                reference_params=(base.reference_params or base.optimi_params).with_(
                    weight_decay=1e-5 if base.fully_decoupled_reference else 0.01,
                    decouple_lr=True,
                ),
            )
        )
    return out


def discover_tests(root: Path | None = None) -> list[OptTest]:
    """
    Discover `opt_*.py` modules in this package. Accept exactly:
      - TESTS: list[OptTest]
      - BASE: OptTest -> expanded via default_variants(BASE)
    """
    if root is None:
        root = Path(__file__).parent
    cases: list[OptTest] = []
    for f in root.glob("opt_*.py"):
        mod = importlib.import_module(f".{f.stem}", package=__package__)
        if hasattr(mod, "TESTS"):
            cases.extend(getattr(mod, "TESTS"))
        elif hasattr(mod, "BASE"):
            base = getattr(mod, "BASE")
            cases.extend(default_variants(base))
    return cases


def optimizer_names() -> list[str]:
    return sorted({c.optimizer_name for c in discover_tests()})
