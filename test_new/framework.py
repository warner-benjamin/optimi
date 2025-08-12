"""Core framework components for the unified optimizer test system.

This module provides the foundational dataclasses and utilities for defining
and executing optimizer tests in a type-safe, self-contained manner.
"""

import inspect
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from optimi.optimizer import OptimiOptimizer
from torch.optim.optimizer import Optimizer


@dataclass
class ToleranceConfig:
    """Tolerance configuration for numerical comparisons."""

    atol: float = 1e-6
    rtol: float = 1e-5
    max_error_rate: float = 0.0005
    equal_nan: bool = False


@dataclass
class BaseParams:
    """Base class for all optimizer parameters with common fields."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    decouple_wd: bool = False
    decouple_lr: bool = False
    triton: bool = False

    def _filter_kwargs_for_class(self, optimizer_class: type) -> dict[str, Any]:
        """Filter parameters based on optimizer signature inspection."""
        if optimizer_class is None:
            return {}

        # Get the optimizer's __init__ signature
        sig = inspect.signature(optimizer_class.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}

        # Filter our parameters to only include those accepted by the optimizer
        return {k: v for k, v in asdict(self).items() if k in valid_params}

    def to_optimi_kwargs(self, optimi_class: type) -> dict[str, Any]:
        """Convert to kwargs for optimi optimizer."""
        return self._filter_kwargs_for_class(optimi_class)

    def to_reference_kwargs(self, reference_class: type) -> dict[str, Any]:
        """Convert to kwargs for reference optimizer."""
        return self._filter_kwargs_for_class(reference_class)


@dataclass
class OptimizerTest:
    """Complete self-contained optimizer test case."""

    # Test identification
    name: str  # "adam_base", "sgd_momentum", etc.

    # Optimizer classes and parameters
    optimi_class: type[OptimiOptimizer]
    optimi_params: BaseParams
    reference_class: type[Optimizer]
    reference_params: BaseParams | None = None

    # Optional fully decoupled reference
    fully_decoupled_reference: Optimizer | None = None

    # Test behavior overrides (optional)
    test_decoupled_wd: bool = True
    skip_tests: list[str] = field(default_factory=list)
    any_precision: bool = False
    custom_iterations: dict[str, int] | None = None
    custom_tolerances: dict[torch.dtype, ToleranceConfig] | None = None
    # Optional constraints
    only_dtypes: list[torch.dtype] | None = None

    def __post_init__(self):
        """Post-initialization checks and adjustments."""
        if self.reference_params is None:
            self.reference_params = deepcopy(self.optimi_params)

        if self.custom_tolerances is None:
            self.custom_tolerances = {}
        if self.custom_tolerances.get(torch.float32, None) is None:
            self.custom_tolerances[torch.float32] = ToleranceConfig()
        if self.custom_tolerances.get(torch.bfloat16, None) is None:
            self.custom_tolerances[torch.bfloat16] = ToleranceConfig(atol=1e-3, rtol=1e-2, max_error_rate=0.01)
        if self.custom_tolerances.get(torch.float16, None) is None:
            self.custom_tolerances[torch.float16] = ToleranceConfig(atol=1e-4, rtol=1e-3, max_error_rate=0.01)

    @property
    def optimizer_name(self) -> str:
        """Extract optimizer name from test name (e.g., 'adam' from 'adam_base')."""
        return self.name.split("_")[0] if "_" in self.name else self.name

    @property
    def variant_name(self) -> str:
        """Extract variant name from test name (e.g., 'base' from 'adam_base')."""
        parts = self.name.split("_", 1)
        return parts[1] if len(parts) > 1 else "base"

    def to_optimi_kwargs(self) -> dict[str, Any]:
        """Get kwargs for optimi optimizer."""
        return self.optimi_params.to_optimi_kwargs(self.optimi_class)

    def to_reference_kwargs(self) -> dict[str, Any]:
        """Get kwargs for reference optimizer."""
        return self.reference_params.to_reference_kwargs(self.reference_class)

    def should_skip_test(self, test_type: str) -> bool:
        """Check if a specific test type should be skipped."""
        return test_type in self.skip_tests

    def get_tolerance(self, dtype: torch.dtype) -> ToleranceConfig:
        """Get tolerance configuration for specific dtype."""
        return self.custom_tolerances[dtype]

    # Backwards-compatible alias to support existing call sites
    def get_tolerance_for_dtype(self, dtype: torch.dtype) -> ToleranceConfig:
        """Backward-compatible alias for get_tolerance."""
        return self.get_tolerance(dtype)

    def get_iterations_for_test(self, test_type: str) -> int:
        """Get number of iterations for specific test type."""
        if self.custom_iterations and test_type in self.custom_iterations:
            return self.custom_iterations[test_type]

        # Default iterations based on test type
        defaults = {"correctness": 10, "gradient_release": 5, "accumulation": 5}
        return defaults.get(test_type, 10)

    def supports_l2_weight_decay(self) -> bool:
        """Check if optimizer supports L2 weight decay."""
        # all optimi optimizers which support l2 weight decay have a decouple_wd parameter
        return "decouple_wd" in inspect.signature(self.optimi_class.__init__).parameters


def assert_most_approx_close(
    a: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    max_error_count: int = 0,
    max_error_rate: float | None = None,
    name: str = "",
) -> None:
    """Assert that most values in two tensors are approximately close.

    Allows for a small number of errors based on max_error_count and max_error_rate.
    """
    idx = torch.isclose(a.float(), b.float(), rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()

    if max_error_rate is not None:
        if error_count > (a.numel()) * max_error_rate and error_count > max_error_count:
            print(f"{name}Too many values not close: assert {error_count} < {(a.numel()) * max_error_rate}")
            torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol)
    elif error_count > max_error_count:
        print(f"{name}Too many values not close: assert {error_count} < {max_error_count}")
        torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol)
