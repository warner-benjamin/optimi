import pytest

def pytest_addoption(parser):
    """Add command-line option to specify a single GPU"""
    parser.addoption(
        "--gpu-id",
        action="store",
        type=int,
        default=None,
        help="Specify a single GPU to use (e.g. --gpu-id=0)"
    )