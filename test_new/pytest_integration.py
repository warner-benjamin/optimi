"""Pytest integration functions for automatic mark generation.

This module provides functions to create pytest parameters with automatic marks
for optimizers, devices, dtypes, and backends, enabling flexible test execution.
"""

import pytest
import torch

from .optimizer_tests import ALL_OPTIMIZER_TESTS, _initialize_test_registry

_CACHED_TESTS: list | None = None


def create_marked_optimizer_tests():
    """Create optimizer test parameters with automatic marks.

    Each test gets marked with its optimizer name for targeted test execution.

    Returns:
        list[pytest.param]: List of pytest parameters with optimizer marks.
    """
    # Call discover_optimizer_tests directly to avoid global variable issues
    from .optimizer_tests import discover_optimizer_tests

    global _CACHED_TESTS
    if _CACHED_TESTS is None:
        _CACHED_TESTS = discover_optimizer_tests()
    return [pytest.param(test, marks=pytest.mark.__getattr__(test.optimizer_name), id=test.name) for test in _CACHED_TESTS]


def create_marked_device_types():
    """Create device type parameters with marks.

    Returns:
        list[pytest.param]: List of device parameters with device marks.
    """
    return [
        pytest.param("cpu", marks=pytest.mark.cpu, id="cpu"),
        pytest.param("gpu", marks=pytest.mark.gpu, id="gpu"),
    ]


def create_marked_dtypes():
    """Create dtype parameters with marks.

    Only includes float32 and bfloat16 as specified in requirements.

    Returns:
        list[pytest.param]: List of dtype parameters with dtype marks.
    """
    return [
        pytest.param(torch.float32, marks=pytest.mark.float32, id="float32"),
        pytest.param(torch.bfloat16, marks=pytest.mark.bfloat16, id="bfloat16"),
    ]


def create_marked_backends():
    """Create backend parameters with marks.

    Only includes torch and triton backends as specified in requirements.

    Returns:
        list[pytest.param]: List of backend parameters with backend marks.
    """
    return [
        pytest.param("torch", marks=pytest.mark.torch, id="torch"),
        pytest.param("triton", marks=pytest.mark.triton, id="triton"),
    ]


def create_gpu_only_device_types():
    """Create device type parameters for GPU-only tests.

    Returns:
        list[pytest.param]: List containing only GPU device parameter.
    """
    return [pytest.param("gpu", marks=pytest.mark.gpu, id="gpu")]


def create_float32_only_dtypes():
    """Create dtype parameters for float32-only tests.

    Returns:
        list[pytest.param]: List containing only float32 dtype parameter.
    """
    return [pytest.param(torch.float32, marks=pytest.mark.float32, id="float32")]


def get_optimizer_marks():
    """Get all available optimizer marks.

    Returns:
        list[str]: List of optimizer names that can be used as pytest marks.
    """
    _initialize_test_registry()
    return sorted({test.optimizer_name for test in ALL_OPTIMIZER_TESTS})


def get_device_marks():
    """Get all available device marks.

    Returns:
        list[str]: List of device names that can be used as pytest marks.
    """
    return ["cpu", "gpu"]


def get_dtype_marks():
    """Get all available dtype marks.

    Returns:
        list[str]: List of dtype names that can be used as pytest marks.
    """
    return ["float32", "bfloat16"]


def get_backend_marks():
    """Get all available backend marks.

    Returns:
        list[str]: List of backend names that can be used as pytest marks.
    """
    return ["torch", "triton"]


def create_test_matrix():
    """Create the complete test matrix for all combinations.

    Returns:
        dict: Dictionary containing all parameter combinations for testing.
    """
    return {
        "optimizer_tests": create_marked_optimizer_tests(),
        "device_types": create_marked_device_types(),
        "dtypes": create_marked_dtypes(),
        "backends": create_marked_backends(),
        "gpu_only_devices": create_gpu_only_device_types(),
        "float32_only_dtypes": create_float32_only_dtypes(),
    }


def print_mark_summary():
    """Print a summary of available marks for debugging."""
    print("Available pytest marks:")
    print(f"  Optimizers: {', '.join(get_optimizer_marks())}")
    print(f"  Devices: {', '.join(get_device_marks())}")
    print(f"  Dtypes: {', '.join(get_dtype_marks())}")
    print(f"  Backends: {', '.join(get_backend_marks())}")

    print(
        f"\nTotal test combinations: {len(create_marked_optimizer_tests())} optimizers × "
        f"{len(get_device_marks())} devices × {len(get_dtype_marks())} dtypes × "
        f"{len(get_backend_marks())} backends"
    )
