# Optimizer utilities

import torch
from packaging.version import parse

MIN_TORCH_2_1 = parse(torch.__version__) >= parse("2.1")


def debias(beta: float, step: int) -> float:
    """Adam-style debias correction. Returns `1 - beta ** step`."""
    return 1 - beta**step


def debias_beta(beta: float, step: int) -> float:
    """Applies the Adam-style debias correction into beta.

    Simplified version of `betahat = beta*(1-beta**(step-1))/(1-beta**step)`
    """
    return (beta**step - beta) / (beta**step - 1)
