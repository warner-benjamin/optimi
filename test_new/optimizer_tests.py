"""Automatic test discovery and generation system for optimizer tests.

This module provides functionality to automatically discover optimizer test definitions
from test files and generate torch test variants, creating a comprehensive test registry.
"""

import importlib
import warnings
from pathlib import Path
from typing import Any

from .framework import OptimizerTest


def discover_optimizer_tests() -> list[OptimizerTest]:
    """Automatically discover and generate tests from all test modules.

    Scans the test_new directory for Python files containing optimizer test definitions.
    Each module should define either BASE_TEST (for auto-generation) or ALL_TESTS (custom).

    Returns:
        List of all discovered and generated OptimizerTest instances.
    """
    all_tests = []
    test_dir = Path(__file__).parent
    test_files = [f for f in test_dir.glob("opt_*.py") if f.is_file()]

    for test_file in test_files:
        module_name = test_file.stem
        try:
            # Import the test module using package-relative import
            # __package__ will be 'optimi.test_new' when this file is imported as a package module
            module = importlib.import_module(f".{module_name}", package=__package__)

            # Check for ALL_TESTS first (custom test definitions)
            if hasattr(module, "ALL_TESTS"):
                tests = getattr(module, "ALL_TESTS")
                if isinstance(tests, list) and all(isinstance(t, OptimizerTest) for t in tests):
                    all_tests.extend(tests)
                else:
                    warnings.warn(f"Module {module_name} has ALL_TESTS but it's not a list of OptimizerTest instances")

            # Check for BASE_TEST (for auto-generation)
            elif hasattr(module, "BASE_TEST"):
                base_test = getattr(module, "BASE_TEST")
                if isinstance(base_test, OptimizerTest):
                    # Generate torch variants from base test
                    generated_tests = auto_generate_variants(base_test)
                    all_tests.extend(generated_tests)
                else:
                    warnings.warn(f"Module {module_name} has BASE_TEST but it's not an OptimizerTest instance")

            else:
                warnings.warn(f"Module {module_name} has neither BASE_TEST nor ALL_TESTS defined")

        except ImportError as e:
            warnings.warn(f"Failed to import test module {module_name}: {e}")
        except Exception as e:
            warnings.warn(f"Error processing test module {module_name}: {e}")

    return all_tests


def auto_generate_variants(base_test: OptimizerTest) -> list[OptimizerTest]:
    """Automatically generate torch test variants from a base test.

    Generates the following torch variants:
    - base: Original test with no weight decay
    - weight_decay: Test with weight decay enabled
    - decoupled_wd: Test with decoupled weight decay
    - decoupled_lr: Test with decoupled learning rate

    Args:
        base_test: Base OptimizerTest to generate variants from.

    Returns:
        List of generated test variants.
    """
    variants = []

    # Base variant (ensure weight_decay is 0)
    base_params = _copy_params_with_overrides(base_test.optimi_params, weight_decay=0.0)
    base_ref_params = _copy_params_with_overrides(base_test.reference_params, weight_decay=0.0)

    base_variant = OptimizerTest(
        name=f"{base_test.optimizer_name}_base",
        optimi_class=base_test.optimi_class,
        optimi_params=base_params,
        reference_class=base_test.reference_class,
        reference_params=base_ref_params,
        skip_tests=base_test.skip_tests.copy(),
        only_dtypes=base_test.only_dtypes,
        any_precision=base_test.any_precision,
        custom_iterations=base_test.custom_iterations,
        custom_tolerances=base_test.custom_tolerances,
    )
    variants.append(base_variant)

    # L2 weight decay variant
    if base_test.supports_l2_weight_decay():
        l2_params = _copy_params_with_overrides(base_test.optimi_params, weight_decay=0.01)
        l2_ref_params = _copy_params_with_overrides(base_test.reference_params, weight_decay=0.01)

        l2_variant = OptimizerTest(
            name=f"{base_test.optimizer_name}_l2_wd",
            optimi_class=base_test.optimi_class,
            optimi_params=l2_params,
            reference_class=base_test.reference_class,
            reference_params=l2_ref_params,
            skip_tests=base_test.skip_tests.copy(),
            only_dtypes=base_test.only_dtypes,
            any_precision=base_test.any_precision,
            custom_iterations=base_test.custom_iterations,
            custom_tolerances=base_test.custom_tolerances,
        )
        variants.append(l2_variant)

    # Decoupled weight decay variant
    if base_test.test_decoupled_wd:
        decoupled_wd_params = _copy_params_with_overrides(base_test.optimi_params, weight_decay=0.01, decouple_wd=True)
        decoupled_wd_ref_params = _copy_params_with_overrides(base_test.reference_params, weight_decay=0.01, decouple_wd=True)

        decoupled_wd_variant = OptimizerTest(
            name=f"{base_test.optimizer_name}_decoupled_wd",
            optimi_class=base_test.optimi_class,
            optimi_params=decoupled_wd_params,
            reference_class=base_test.reference_class,
            reference_params=decoupled_wd_ref_params,
            skip_tests=base_test.skip_tests.copy(),
            only_dtypes=base_test.only_dtypes,
            any_precision=base_test.any_precision,
            custom_iterations=base_test.custom_iterations,
            custom_tolerances=base_test.custom_tolerances,
        )
        variants.append(decoupled_wd_variant)

        # Decoupled learning rate variant
        decoupled_lr_params = _copy_params_with_overrides(base_test.optimi_params, weight_decay=1e-5, decouple_lr=True)
        if base_test.fully_decoupled_reference is not None:
            decoupled_lr_ref_params = _copy_params_with_overrides(base_test.reference_params, weight_decay=1e-5, decouple_lr=True)
            reference_class = base_test.fully_decoupled_reference
        else:
            decoupled_lr_ref_params = _copy_params_with_overrides(base_test.reference_params, weight_decay=0.01, decouple_lr=True)
            reference_class = base_test.reference_class

        decoupled_lr_variant = OptimizerTest(
            name=f"{base_test.optimizer_name}_decoupled_lr",
            optimi_class=base_test.optimi_class,
            optimi_params=decoupled_lr_params,
            reference_class=reference_class,
            reference_params=decoupled_lr_ref_params,
            skip_tests=base_test.skip_tests.copy(),
            only_dtypes=base_test.only_dtypes,
            any_precision=base_test.any_precision,
            custom_iterations=base_test.custom_iterations,
            custom_tolerances=base_test.custom_tolerances,
        )
        variants.append(decoupled_lr_variant)

    return variants


def _copy_params_with_overrides(params: Any, **overrides: Any) -> Any:
    """Create a copy of parameter dataclass with specified overrides.

    Args:
        params: Original parameter dataclass instance.
        **overrides: Field values to override.

    Returns:
        New parameter instance with overrides applied.
    """
    # Get all current field values
    current_values = {}
    for field_info in params.__dataclass_fields__.values():
        current_values[field_info.name] = getattr(params, field_info.name)

    # Apply overrides
    current_values.update(overrides)

    # Create new instance
    return type(params)(**current_values)


# Central registry of all discovered tests
ALL_OPTIMIZER_TESTS: list[OptimizerTest] = []


def _initialize_test_registry() -> None:
    """Initialize the test registry by discovering all tests."""
    global ALL_OPTIMIZER_TESTS
    if not ALL_OPTIMIZER_TESTS:  # Only initialize once
        ALL_OPTIMIZER_TESTS = discover_optimizer_tests()


def get_tests_by_optimizer(optimizer_name: str) -> list[OptimizerTest]:
    """Get all tests for a specific optimizer.

    Args:
        optimizer_name: Name of the optimizer (e.g., 'adam', 'sgd').

    Returns:
        List of OptimizerTest instances for the specified optimizer.
    """
    _initialize_test_registry()
    return [test for test in ALL_OPTIMIZER_TESTS if test.optimizer_name == optimizer_name]


def get_tests_by_variant(variant_name: str) -> list[OptimizerTest]:
    """Get all tests for a specific variant across all optimizers.

    Args:
        variant_name: Name of the variant (e.g., 'base', 'weight_decay').

    Returns:
        List of OptimizerTest instances for the specified variant.
    """
    _initialize_test_registry()
    return [test for test in ALL_OPTIMIZER_TESTS if test.variant_name == variant_name]


def get_test_by_name(name: str) -> OptimizerTest | None:
    """Get a specific test by its full name.

    Args:
        name: Full test name (e.g., 'adam_base', 'sgd_momentum').

    Returns:
        OptimizerTest instance if found, None otherwise.
    """
    _initialize_test_registry()
    for test in ALL_OPTIMIZER_TESTS:
        if test.name == name:
            return test
    return None


def get_all_optimizer_names() -> list[str]:
    """Get list of all available optimizer names.

    Returns:
        Sorted list of unique optimizer names.
    """
    _initialize_test_registry()
    optimizer_names = {test.optimizer_name for test in ALL_OPTIMIZER_TESTS}
    return sorted(optimizer_names)


def get_all_variant_names() -> list[str]:
    """Get list of all available variant names.

    Returns:
        Sorted list of unique variant names.
    """
    _initialize_test_registry()
    variant_names = {test.variant_name for test in ALL_OPTIMIZER_TESTS}
    return sorted(variant_names)


def get_test_count() -> int:
    """Get total number of discovered tests.

    Returns:
        Total count of OptimizerTest instances.
    """
    _initialize_test_registry()
    return len(ALL_OPTIMIZER_TESTS)


def print_test_summary() -> None:
    """Print a summary of discovered tests for debugging."""
    _initialize_test_registry()

    print(f"Discovered {len(ALL_OPTIMIZER_TESTS)} optimizer tests:")
    print(f"Optimizers: {', '.join(get_all_optimizer_names())}")
    print(f"Variants: {', '.join(get_all_variant_names())}")

    # Group by optimizer
    for optimizer_name in get_all_optimizer_names():
        tests = get_tests_by_optimizer(optimizer_name)
        test_names = [test.name for test in tests]
        print(f"  {optimizer_name}: {', '.join(test_names)}")
