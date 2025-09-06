# Copyright (c) 2023-present Benjamin Warner
# SPDX-License-Identifier: MIT

import inspect
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import torch
from packaging.version import parse
from torch import nn
from torch.utils._foreach_utils import _foreach_supported_types

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

MIN_TORCH_2_1 = parse(torch.__version__) >= parse("2.1")
MIN_TORCH_2_6 = parse(torch.__version__) >= parse("2.6")
HAS_TRITON = HAS_TRITON and MIN_TORCH_2_6


def debias(beta: float, step: int) -> float:
    """Adam-style debias correction. Returns `1 - beta ** step`."""
    return 1 - beta**step


def debias_beta(beta: float, step: int) -> float:
    """Applies the Adam-style debias correction into beta.

    Simplified version of `beta_hat = beta*(1-beta**(step-1))/(1-beta**step)`
    """
    return (beta**step - beta) / (beta**step - 1)


# modified from timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/optim_factory.py
# Copyright 2019 Ross Wightman, Apache-2.0 License
def param_groups_weight_decay(
    model: nn.Module, weight_decay: float = 1e-2, additional_layers: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """Creates parameter groups excluding bias and normalization layers from weight decay.

    Parameters:
        model: PyTorch model to create parameter groups for
        weight_decay: Weight decay coefficient applied to eligible parameters (default: 1e-2)
        additional_layers: Iterable of layer name substrings to exclude from weight decay.
            Any parameter whose name contains one of these substrings will be excluded from
            weight decay.

    Returns:
        List of two parameter group dictionaries, one with and one without weight decay.
    """
    additional_layers = set(additional_layers) if additional_layers is not None else set()
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or any(n in name for n in additional_layers):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def _normalize_buffers(
    fp32_buffers: str | type[nn.Module] | Iterable[str] | Iterable[type[nn.Module]] | None,
) -> tuple[set[str], tuple[type[nn.Module], ...]]:
    if fp32_buffers is None:
        return set(), tuple()
    if not isinstance(fp32_buffers, Iterable):
        fp32_buffers = (fp32_buffers,)

    keywords = set()
    types = set()
    for buf in fp32_buffers:
        if isinstance(buf, str):
            keywords.add(buf.lower())
        elif isinstance(buf, type) and issubclass(buf, nn.Module):
            types.add(buf)
        else:
            raise TypeError("fp32_buffers items must be str or a nn.Module subclass type")
    return keywords, tuple(types)


def _normalize_modules(fp32_modules: type[nn.Module] | Iterable[type[nn.Module]] | None) -> tuple[type[nn.Module], ...]:
    if fp32_modules is None:
        return tuple()
    if not isinstance(fp32_modules, Iterable):
        fp32_modules = (fp32_modules,)
    if not all(isinstance(t, type) and issubclass(t, nn.Module) for t in fp32_modules):
        raise TypeError("fp32_modules must be nn.Module subclass type(s)")
    return fp32_modules


def to_low_precision(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    fp32_modules: type[nn.Module] | Iterable[type[nn.Module]] | None = nn.Embedding,
    fp32_buffers: str | type[nn.Module] | Iterable[str] | Iterable[type[nn.Module]] | None = ("rope", "rotary"),
) -> nn.Module:
    """Cast model to a low-precision dtype keeping select modules and buffers in float32.

    Keeps all parameters and buffers of ``fp32_modules`` in float32 (default: ``nn.Embedding``).
    Keeps only buffers in float32 for modules whose qualified name contains any keyword in
    ``fp32_buffers`` or whose type is listed there (useful for RoPE-style buffers).
    Casts everything else to ``dtype`` and optionally moves to ``device``.

    Args:
        model: Module to cast in-place.
        dtype: Target dtype (default: ``torch.bfloat16``).
        device: Optional target device (default: None).
        fp32_modules: Modules to keep in float32 (default: ``nn.Embedding``).
        fp32_buffers: Names and/or modules to keep buffers in float32 (default: ("rope", "rotary")).

    Returns:
        The input ``model`` (cast in-place).
    """
    kw, buf_types = _normalize_buffers(fp32_buffers)
    mod_types = _normalize_modules(fp32_modules)

    # Check if PyTorch has the optional 'recurse' parameter on Module._apply
    if "recurse" not in inspect.signature(nn.Module._apply).parameters:
        raise ValueError("`to_low_precision` requires PyTorch version 2.1 or newer.")

    def recurse(m: nn.Module, qual: str = ""):
        # Decide policy for this module
        keep_params_fp32 = isinstance(m, mod_types)
        keep_buffers_fp32 = keep_params_fp32 or any(k in qual for k in kw) or isinstance(m, buf_types)

        # Snapshot local param / buffer ids so our *_apply fn can tell them apart
        param_ids = {id(p) for _, p in m.named_parameters(recurse=False)}
        buffer_ids = {id(b) for _, b in m.named_buffers(recurse=False)}

        # Per-module param cast (buffers are returned unchanged)
        def cast_params(t: torch.Tensor):
            if id(t) in param_ids and t.is_floating_point():
                cast_type = torch.float32 if keep_params_fp32 else dtype
                return t.to(dtype=cast_type, device=device)
            return t

        # Per-module buffer cast (params are returned unchanged)
        def cast_buffers(t: torch.Tensor):
            if id(t) in buffer_ids and t.is_floating_point():
                cast_type = torch.float32 if keep_buffers_fp32 else dtype
                return t.to(dtype=cast_type, device=device)
            return t

        # Use PyTorch's own _apply machinery (handles swap/set_data/grad properly)
        m._apply(cast_params, recurse=False)
        m._apply(cast_buffers, recurse=False)

        # Recurse with qualified names
        for child_name, child in m.named_children():
            child_qual = f"{qual}.{child_name}" if qual else child_name
            recurse(child, child_qual)

    recurse(model)

    return model


def _device_guard(tensor: torch.Tensor):
    """Returns context manager to ensure that the Triton kernel launches on the correct device."""
    if tensor.is_cuda:  # NVIDIA or AMD/ROCm
        return torch.cuda.device_of(tensor)
    elif hasattr(tensor, "is_xpu") and tensor.is_xpu:  # Intel GPUs
        return torch.xpu.device_of(tensor)
    else:  # CPU or other back-ends
        return nullcontext()


def _triton_kernels_supported_device(tensor: torch.Tensor):
    if tensor.is_cuda:  # NVIDIA or AMD/ROCm
        return torch.cuda.is_bf16_supported()
    elif hasattr(tensor, "is_xpu") and tensor.is_xpu:  # Intel GPUs
        return torch.xpu.is_bf16_supported()
    else:
        return False


# modified from PyTorch's _default_to_fused_or_foreach
# Copyright 2013-present PyTorch contributors, PyTorch BSD-style license
def _default_to_triton(params: list[torch.Tensor]) -> tuple[bool, bool]:
    if torch.jit.is_scripting():
        return False

    triton = HAS_TRITON and all(
        p is None or (type(p) in _foreach_supported_types and _triton_kernels_supported_device(p) and torch.is_floating_point(p))
        for p in params
    )
    return triton


def _get_triton_block_size(n_elements: int) -> int:
    """Returns the Triton block size based on the number of elements."""
    if n_elements < 2048:
        return 128
    elif n_elements < 4096:
        return 256
    elif n_elements < 8192:
        return 512
    else:
        return 1024


TORCH_TO_TRITON_DTYPE = {}

if HAS_TRITON:
    import triton.language as tl
    from torch._inductor.utils import triton_type_to_torch

    for triton_dtype in tl.dtype.SINT_TYPES + tl.dtype.UINT_TYPES + tl.dtype.FP_TYPES:
        try:
            triton_type = triton_dtype.replace("fp", "float").replace("bf", "bfloat")
            torch_type = triton_type_to_torch(triton_type)
            TORCH_TO_TRITON_DTYPE[torch_type] = tl.dtype(triton_dtype)
        except AttributeError:
            try:
                # Handle the case where triton_type_to_torch might not recognize the Triton type
                torch_type = triton_type_to_torch(f"tl.{triton_type}")
                TORCH_TO_TRITON_DTYPE[torch_type] = tl.dtype(triton_dtype)
            except AttributeError:
                pass
