import torch

try:
    import triton
except ImportError:
    triton = None


def pytest_report_header(config):
    if triton is None:
        return f"libaries: PyTorch {torch.__version__}"
    else:
        return f"libaries: PyTorch {torch.__version__}, Triton: {triton.__version__}"


def pytest_addoption(parser):
    """Add command-line option to specify a single GPU"""
    parser.addoption("--gpu-id", action="store", type=int, default=None, help="Specify a single GPU to use (e.g. --gpu-id=0)")
