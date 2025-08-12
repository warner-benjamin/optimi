"""Pytest configuration and fixtures for the unified optimizer test framework.

This module provides pytest configuration, custom mark registration, and fixtures
for running optimizer tests across different devices, dtypes, and backends.
"""

import pytest
import torch
from packaging import version

from .optimizer_tests import get_all_optimizer_names


def pytest_configure(config):
    """Configure pytest with custom marks for optimizer testing."""

    # Register device marks
    config.addinivalue_line("markers", "cpu: mark test to run on CPU")
    config.addinivalue_line("markers", "gpu: mark test to run on GPU")

    # Register dtype marks
    config.addinivalue_line("markers", "float32: mark test to run with float32 dtype")
    config.addinivalue_line("markers", "bfloat16: mark test to run with bfloat16 dtype")

    # Register backend marks
    config.addinivalue_line("markers", "torch: mark test to run with torch backend")
    config.addinivalue_line("markers", "triton: mark test to run with triton backend")

    optimizer_names = get_all_optimizer_names()
    for optimizer_name in optimizer_names:
        config.addinivalue_line("markers", f"{optimizer_name}: mark test for {optimizer_name} optimizer")


# Check for minimum PyTorch version for Triton support
MIN_TORCH_2_6 = version.parse("2.6.0")
CURRENT_TORCH_VERSION = version.parse(torch.__version__.split("+")[0])  # Remove any +cu118 suffix
HAS_TRITON_SUPPORT = CURRENT_TORCH_VERSION >= MIN_TORCH_2_6


@pytest.fixture(scope="session")
def gpu_device():
    """Provide GPU device for testing if available.

    Returns:
        torch.device: GPU device (cuda, xpu, or mps) if available, otherwise None.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return None


@pytest.fixture(scope="session")
def has_gpu(gpu_device):
    """Check if GPU is available for testing.

    Returns:
        bool: True if GPU device is available, False otherwise.
    """
    return gpu_device is not None


@pytest.fixture(scope="session")
def has_triton():
    """Check if Triton backend is available.

    Returns:
        bool: True if Triton is supported (PyTorch >= 2.6), False otherwise.
    """
    return HAS_TRITON_SUPPORT


@pytest.fixture
def tolerance_config():
    """Provide default tolerance configuration for numerical comparisons.

    Returns:
        dict: Default tolerance settings for different dtypes.
    """
    from .framework import ToleranceConfig

    return {
        torch.float32: ToleranceConfig(rtol=1e-5, atol=1e-8),
        torch.bfloat16: ToleranceConfig(rtol=1e-3, atol=1e-5),  # More relaxed for bfloat16
    }


@pytest.fixture
def cpu_device():
    """Provide CPU device for testing.

    Returns:
        torch.device: CPU device.
    """
    return torch.device("cpu")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic skipping for unavailable resources."""

    # Skip GPU tests if no GPU is available
    if not torch.cuda.is_available() and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Skip Triton tests if not supported
    if not HAS_TRITON_SUPPORT:
        skip_triton = pytest.mark.skip(reason=f"Triton requires PyTorch >= {MIN_TORCH_2_6}, got {CURRENT_TORCH_VERSION}")
        for item in items:
            if "triton" in item.keywords:
                item.add_marker(skip_triton)


def pytest_runtest_setup(item):
    """Setup hook to perform additional test skipping based on marks."""

    # Skip GPU tests on CPU-only systems
    if "gpu" in item.keywords:
        gpu_available = (
            torch.cuda.is_available()
            or (hasattr(torch, "xpu") and torch.xpu.is_available())
            or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        )
        if not gpu_available:
            pytest.skip("GPU not available")

    # Skip Triton tests if not supported
    if "triton" in item.keywords and not HAS_TRITON_SUPPORT:
        pytest.skip(f"Triton requires PyTorch >= {MIN_TORCH_2_6}, got {CURRENT_TORCH_VERSION}")
