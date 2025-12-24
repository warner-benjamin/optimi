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


class TestType(Enum):
    correctness = "correctness"
    gradient_release = "gradient_release"
    accumulation = "accumulation"


class DeviceType(Enum):
    cpu = "cpu"
    gpu = "gpu"


class Backend(Enum):
    torch = "torch"
    triton = "triton"
    foreach = "foreach"


@dataclass
class Tolerance:
    atol: float = 1e-6
    rtol: float = 1e-5
    max_error_rate: float = 5e-4
    equal_nan: bool = False


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
    test_decoupled_wd: bool = True
    skip_tests: list[TestType] = field(default_factory=list)
    any_precision: bool = False
    custom_iterations: dict[TestType | tuple[TestType, DeviceType], int] | None = None
    custom_tolerances: dict[torch.dtype, Tolerance] | None = None
    only_dtypes: list[torch.dtype] | None = None

    def __post_init__(self):
        if self.reference_params is None:
            self.reference_params = self.optimi_params
        if self.custom_tolerances is None:
            self.custom_tolerances = {}
        # reasonable defaults; override per-case as needed
        self.custom_tolerances.setdefault(torch.float32, Tolerance())
        self.custom_tolerances.setdefault(torch.bfloat16, Tolerance(atol=1e-3, rtol=1e-2, max_error_rate=0.01))
        self.custom_tolerances.setdefault(torch.float16, Tolerance(atol=1e-4, rtol=1e-3, max_error_rate=0.01))

    @property
    def optimizer_name(self) -> str:
        return self.name.split("_", 1)[0]

    @property
    def variant_name(self) -> str:
        return self.name.split("_", 1)[1] if "_" in self.name else "base"

    def to_optimi_kwargs(self, backend: Backend | None = None) -> dict[str, Any]:
        # Both new BaseParams and legacy test_new.framework.BaseParams expose this
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
        custom_tolerances=base.custom_tolerances,
        only_dtypes=base.only_dtypes,
        fully_decoupled_reference=base.fully_decoupled_reference,
    )
    out.append(base_test)

    # L2 weight decay if optimizer supports decouple_wd arg
    if "decouple_wd" in inspect.signature(base.optimi_class.__init__).parameters:
        out.append(
            replace(
                base_test,
                name=f"{base.optimizer_name}_l2_wd",
                optimi_params=base.optimi_params.with_(weight_decay=0.01, decouple_wd=False),
                reference_params=(base.reference_params or base.optimi_params).with_(weight_decay=0.01, decouple_wd=False),
            )
        )

    # Decoupled weight decay
    if base.test_decoupled_wd:
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
