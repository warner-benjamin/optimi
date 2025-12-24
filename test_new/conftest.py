"""Pytest configuration and fixtures for the unified optimizer test framework.

This module provides pytest configuration, custom mark registration, and the
`gpu_device` fixture used by tests.
"""

import pytest
import torch

from .config import optimizer_names


def pytest_configure(config):
    "Configure pytest with custom marks for optimizer testing."
    # Register device marks
    config.addinivalue_line("markers", "cpu: mark test to run on CPU")
    config.addinivalue_line("markers", "gpu: mark test to run on GPU")

    # Register dtype marks
    config.addinivalue_line("markers", "float32: mark test to run with float32 dtype")
    config.addinivalue_line("markers", "bfloat16: mark test to run with bfloat16 dtype")

    # Register backend marks
    config.addinivalue_line("markers", "torch: mark test to run with torch backend")
    config.addinivalue_line("markers", "triton: mark test to run with triton backend")

    # Per-optimizer marks (e.g., -m adam, -m sgd)
    for opt_name in optimizer_names():
        config.addinivalue_line("markers", f"{opt_name}: mark test for {opt_name} optimizer")


def pytest_addoption(parser):
    "Add command-line option to specify a single GPU."
    parser.addoption("--gpu-id", action="store", type=int, default=None, help="Specify a single GPU to use (e.g. --gpu-id=0)")


@pytest.fixture()
def gpu_device(worker_id, request):
    """Map xdist workers to available GPU devices in a round-robin fashion, supporting CUDA (NVIDIA/ROCm) and XPU (Intel) backends.

    Use a single specified GPU if --gpu-id is provided"""

    # Check if specific GPU was requested
    specific_gpu = request.config.getoption("--gpu-id")

    # Determine available GPU backend and device count
    if torch.cuda.is_available():
        backend = "cuda"
        device_count = torch.cuda.device_count()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        backend = "xpu"
        device_count = torch.xpu.device_count()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backend = "mps"
        device_count = 0
    else:
        raise RuntimeError("No GPU backend available")

    if specific_gpu is not None:
        return torch.device(f"{backend}:{specific_gpu}")

    if worker_id == "master":
        return torch.device(backend)

    # If no devices available, return default backend
    if device_count == 0:
        return torch.device(backend)

    # Extract worker number from worker_id (e.g., 'gw6' -> 6)
    worker_num = int(worker_id.replace("gw", ""))

    # Map worker to GPU index using modulo to round-robin
    gpu_idx = (worker_num - 1) % device_count
    return torch.device(f"{backend}:{gpu_idx}")
