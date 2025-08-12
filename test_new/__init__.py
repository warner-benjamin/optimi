"""
Unified optimizer test framework.

This module provides a simplified, dataclass-based approach to optimizer testing
that replaces the complex OptimizerSpec/OptimizerVariant architecture.
"""

__version__ = "1.0.0"

# Core framework components
from .framework import BaseParams, OptimizerTest, ToleranceConfig

# Test discovery and registry
from .optimizer_tests import (
    ALL_OPTIMIZER_TESTS,
    auto_generate_variants,
    discover_optimizer_tests,
    get_all_optimizer_names,
    get_all_variant_names,
    get_test_by_name,
    get_test_count,
    get_tests_by_optimizer,
    get_tests_by_variant,
    print_test_summary,
)

# Pytest integration
from .pytest_integration import (
    create_marked_backends,
    create_marked_device_types,
    create_marked_dtypes,
    create_marked_optimizer_tests,
    create_float32_only_dtypes,
    create_gpu_only_device_types,
    create_test_matrix,
    get_backend_marks,
    get_device_marks,
    get_dtype_marks,
    get_optimizer_marks,
    print_mark_summary,
)

__all__ = [
    # Core framework
    "BaseParams",
    "OptimizerTest",
    "ToleranceConfig",
    # Test discovery
    "ALL_OPTIMIZER_TESTS",
    "auto_generate_variants",
    "discover_optimizer_tests",
    "get_all_optimizer_names",
    "get_all_variant_names",
    "get_test_by_name",
    "get_test_count",
    "get_tests_by_optimizer",
    "get_tests_by_variant",
    "print_test_summary",
    # Pytest integration
    "create_marked_backends",
    "create_marked_device_types",
    "create_marked_dtypes",
    "create_marked_optimizer_tests",
    "create_float32_only_dtypes",
    "create_gpu_only_device_types",
    "create_test_matrix",
    "get_backend_marks",
    "get_device_marks",
    "get_dtype_marks",
    "get_optimizer_marks",
    "print_mark_summary",
]
