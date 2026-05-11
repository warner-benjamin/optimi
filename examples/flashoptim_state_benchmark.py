"""Microbenchmark FlashOptim state quantization variants."""

from __future__ import annotations

import argparse
import time

import torch

from optimi import SGD, AdamW, Lion


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    raise RuntimeError("FlashOptim state benchmark requires CUDA or XPU.")


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def run_case(optimizer_cls, optimizer_kwargs, flash_kwargs, n_elements: int, warmup: int, steps: int, device: torch.device) -> float:
    param = torch.nn.Parameter(torch.randn(n_elements, device=device, dtype=torch.bfloat16))
    grad = torch.randn_like(param)
    optimizer = optimizer_cls([param], lr=1e-3, triton=True, **optimizer_kwargs, **flash_kwargs)

    for _ in range(warmup):
        param.grad = grad
        optimizer.step()

    _sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        param.grad = grad
        optimizer.step()
    _sync(device)
    return (time.perf_counter() - start) * 1000 / steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=1_000_000)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = _device()
    configs = [
        ("flash=False", {"flash": False, "quantize": False}),
        ("flash=24", {"flash": True, "quantize": False}),
        ("flash=32", {"flash": "32", "quantize": False}),
        ("quantize=True", {"flash": False, "quantize": True}),
        ("flash=24,quantize=True", {"flash": True, "quantize": True}),
        ("flash=24,stochastic=True", {"flash": True, "stochastic_rounding": True}),
    ]
    optimizers = [
        ("adamw", AdamW, {}),
        ("lion", Lion, {}),
        ("sgd", SGD, {}),
        ("sgd_momentum", SGD, {"momentum": 0.9}),
    ]

    torch.manual_seed(0)
    for name, optimizer_cls, optimizer_kwargs in optimizers:
        for label, flash_kwargs in configs:
            ms = run_case(optimizer_cls, optimizer_kwargs, flash_kwargs, args.elements, args.warmup, args.steps, device)
            print(f"{name:12s} {label:23s} {ms:.3f} ms/step")


if __name__ == "__main__":
    main()
