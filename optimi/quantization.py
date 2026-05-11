# Copyright (c) 2023-present Benjamin Warner
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

GROUP_SIZE = 32
SCALE_SENTINEL = 1e-12


@dataclass(frozen=True)
class QuantizedStateSpec:
    """Grouped optimizer-state quantization parameters."""

    signed: bool
    sqrt: bool = False
    softsign: bool = False

    @property
    def dtype(self) -> torch.dtype:
        """Returns the live quantized tensor dtype."""
        return torch.int8 if self.signed else torch.uint8


MOMENTUM_STATE = QuantizedStateSpec(signed=True, softsign=True)
ADAM_VARIANCE_STATE = QuantizedStateSpec(signed=False, sqrt=True)


def quantized_scale_numel(tensor: Tensor) -> int:
    """Returns the number of grouped scale values needed for `tensor`."""
    return (tensor.numel() + GROUP_SIZE - 1) // GROUP_SIZE


def validate_quantized_state_support(params: list[Tensor], grads: list[Tensor], triton: bool, triton_supported: bool):
    """Validates the Triton path requirements for quantized optimizer state."""
    if not triton or not triton_supported:
        raise ValueError("FlashOptim training support requires triton=True")
    for param, grad in zip(params, grads):
        if not param.is_contiguous() or not grad.is_contiguous():
            raise ValueError("FlashOptim training support requires contiguous parameters and gradients.")


def init_quantized_state(param: Tensor, spec: QuantizedStateSpec) -> tuple[Tensor, Tensor]:
    """Initializes valid zero-encoded quantized state and scales."""
    state = torch.zeros_like(param, dtype=spec.dtype, memory_format=torch.preserve_format)
    scales = torch.ones(quantized_scale_numel(param), dtype=torch.float32, device=param.device)
    return state, scales


def _flat_groups(tensor: Tensor) -> Tensor:
    flat = tensor.reshape(-1)
    padding = (-flat.numel()) % GROUP_SIZE
    if padding:
        flat = torch.nn.functional.pad(flat, (0, padding))
    return flat.reshape(-1, GROUP_SIZE)


def dequantize_state(qstate: Tensor, scales: Tensor, spec: QuantizedStateSpec, dtype: torch.dtype | None = None) -> Tensor:
    """Dequantizes grouped optimizer state."""
    if qstate.numel() == 1 and scales.numel() == 0:
        return qstate.to(dtype=dtype or torch.float32)

    groups = _flat_groups(qstate).to(torch.float32)
    scale_groups = scales.to(torch.float32).reshape(-1, 1)
    if spec.signed:
        normalized = groups / 127.0
        if spec.softsign:
            normalized = normalized / (2.0 - normalized.abs())
        value = normalized * scale_groups
    else:
        value = groups / 255.0 * scale_groups
        if spec.sqrt:
            value = value.square()

    value = value.reshape(-1)[: qstate.numel()].reshape_as(qstate)
    return value.to(dtype=dtype) if dtype is not None else value


def quantize_state_(state: Tensor, qstate: Tensor, scales: Tensor, spec: QuantizedStateSpec):
    """Quantizes `state` into grouped `qstate` and `scales` in-place."""
    groups = _flat_groups(state.to(torch.float32))
    if spec.sqrt:
        groups = groups.sqrt()

    scale = groups.abs().amax(dim=1).clamp_min(SCALE_SENTINEL)
    normalized = groups / scale.reshape(-1, 1)
    if spec.signed:
        if spec.softsign:
            normalized = 2.0 * normalized / (1.0 + normalized.abs())
        encoded = (normalized * 127.0).round().clamp(-127, 127).to(torch.int8)
    else:
        encoded = (normalized * 255.0).round().clamp(0, 255).to(torch.uint8)

    qstate.copy_(encoded.reshape(-1)[: qstate.numel()].reshape_as(qstate))
    scales.copy_(scale.to(scales.dtype))


def export_quantized_state_dict(optimizer, specs: dict[str, tuple[str, QuantizedStateSpec]]):
    """Exports dequantized optimizer state and omits live scale tensors."""
    raw_state_dict = Optimizer.state_dict(optimizer)
    state_dict = {
        "state": {param_id: dict(state) for param_id, state in raw_state_dict["state"].items()},
        "param_groups": [dict(group) for group in raw_state_dict["param_groups"]],
    }
    for group, saved_group in zip(optimizer.param_groups, state_dict["param_groups"]):
        for param, param_id in zip(group["params"], saved_group["params"]):
            state = optimizer.state[param]
            saved_state = state_dict["state"].get(param_id)
            if not saved_state:
                continue

            for state_name, (scale_name, spec) in specs.items():
                if state_name in state and scale_name in state:
                    saved_state[state_name] = dequantize_state(state[state_name], state[scale_name], spec)
                    saved_state.pop(scale_name, None)

    return state_dict


def quantize_loaded_state_(optimizer, specs: dict[str, tuple[str, QuantizedStateSpec]]):
    """Re-quantizes loaded public state tensors for quantized parameter groups."""
    for group in optimizer.param_groups:
        if not group.get("quantize", False):
            continue

        for param in group["params"]:
            state = optimizer.state[param]
            for state_name, (scale_name, spec) in specs.items():
                tensor = state.get(state_name)
                if tensor is None:
                    continue
                if tensor.numel() != param.numel():
                    continue

                qstate, scales = init_quantized_state(param, spec)
                quantize_state_(tensor, qstate, scales, spec)
                state[state_name] = qstate
                state[scale_name] = scales


def restore_integer_state_(optimizer, state_dict, state_names: tuple[str, ...]):
    """Restores integer state tensors that PyTorch may cast during load."""
    for group, saved_group in zip(optimizer.param_groups, state_dict["param_groups"]):
        for param, param_id in zip(group["params"], saved_group["params"]):
            saved_state = state_dict["state"].get(param_id)
            if not saved_state:
                continue

            state = optimizer.state[param]
            for state_name in state_names:
                tensor = saved_state.get(state_name)
                if torch.is_tensor(tensor) and tensor.dtype in (torch.int8, torch.int16, torch.uint8):
                    state[state_name] = tensor.to(device=param.device)
