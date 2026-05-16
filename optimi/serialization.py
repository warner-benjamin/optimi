# Copyright (c) 2023-present Benjamin Warner
# SPDX-License-Identifier: MIT

# Implementation of FlashOptim: Optimizers for Memory-Efficient Training
# FlashOptim - Apache-2.0 - Copyright 2026 Databricks, Inc.

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from optimi.adam import _QUANTIZED_STATES as ADAM_QUANTIZED_STATES
from optimi.adam import Adam
from optimi.adamw import AdamW
from optimi.lion import _QUANTIZED_STATES as LION_QUANTIZED_STATES
from optimi.lion import Lion
from optimi.quantization import (
    COMPRESSED_STATE_DICT_KEY,
    GROUP_SIZE,
    QuantizedStateSpec,
    dequantize_state,
    quantized_scale_numel,
)
from optimi.sgd import _QUANTIZED_STATES as SGD_QUANTIZED_STATES
from optimi.sgd import SGD

CHECKPOINT_VERSION = 1
QUANTIZED_SUFFIX = "::quantized"
SCALES_SUFFIX = "::scales"

_SUPPORTED_SPECS = {
    Adam: ADAM_QUANTIZED_STATES,
    AdamW: ADAM_QUANTIZED_STATES,
    Lion: LION_QUANTIZED_STATES,
    SGD: SGD_QUANTIZED_STATES,
}


def _optimizer_specs(optimizer) -> dict[str, tuple[str, QuantizedStateSpec]]:
    for optimizer_type, specs in _SUPPORTED_SPECS.items():
        if isinstance(optimizer, optimizer_type):
            return specs
    raise ValueError(f"Compressed optimi checkpoints are not supported for {optimizer.__class__.__name__}.")


def _clone_state_value(value):
    return value.detach().clone() if torch.is_tensor(value) else value


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype)


def _spec_metadata(scale_name: str, spec: QuantizedStateSpec, qstate: Tensor, scales: Tensor) -> dict:
    return {
        "scale_name": scale_name,
        "group_size": GROUP_SIZE,
        "signed": spec.signed,
        "sqrt": spec.sqrt,
        "softsign": spec.softsign,
        "dtype": _dtype_name(qstate.dtype),
        "scale_dtype": _dtype_name(scales.dtype),
    }


def _flash_metadata(param: Tensor, flash_comp: Tensor) -> dict:
    if flash_comp.dtype == torch.int8:
        precision = "24"
    elif flash_comp.dtype == torch.int16:
        precision = "32"
    else:
        raise ValueError(f"Invalid flash_comp dtype in optimizer state: {flash_comp.dtype}.")
    return {
        "precision": precision,
        "dtype": _dtype_name(flash_comp.dtype),
        "param_dtype": _dtype_name(param.dtype),
    }


def _stochastic_metadata(seed: Tensor) -> dict:
    return {"dtype": _dtype_name(seed.dtype), "shape": tuple(seed.shape)}


def compressed_state_dict(optimizer) -> dict:
    """Returns an optimi compressed checkpoint for FlashOptim quantized state."""
    specs = _optimizer_specs(optimizer)
    raw_state_dict = Optimizer.state_dict(optimizer)
    state_dict = {
        "state": {param_id: dict(state) for param_id, state in raw_state_dict["state"].items()},
        "param_groups": [dict(group) for group in raw_state_dict["param_groups"]],
    }
    metadata = {
        "version": CHECKPOINT_VERSION,
        "optimizer": optimizer.__class__.__name__,
        "params": {},
    }

    for group, saved_group in zip(optimizer.param_groups, state_dict["param_groups"]):
        for param, param_id in zip(group["params"], saved_group["params"]):
            state = optimizer.state[param]
            saved_state = state_dict["state"].get(param_id)
            if not saved_state:
                continue

            param_metadata = {"param_dtype": _dtype_name(param.dtype), "quantized_states": {}}
            for state_name, (scale_name, spec) in specs.items():
                if state_name not in state or scale_name not in state:
                    saved_state.pop(scale_name, None)
                    continue
                qstate = state[state_name]
                scales = state[scale_name]
                saved_state[f"{state_name}{QUANTIZED_SUFFIX}"] = _clone_state_value(qstate)
                saved_state[f"{state_name}{SCALES_SUFFIX}"] = _clone_state_value(scales)
                saved_state.pop(state_name, None)
                saved_state.pop(scale_name, None)
                param_metadata["quantized_states"][state_name] = _spec_metadata(scale_name, spec, qstate, scales)

            flash_comp = state.get("flash_comp")
            if torch.is_tensor(flash_comp):
                saved_state["flash_comp"] = _clone_state_value(flash_comp)
                param_metadata["flash"] = _flash_metadata(param, flash_comp)

            stochastic_seed = state.get("stochastic_seed")
            if torch.is_tensor(stochastic_seed):
                saved_state["stochastic_seed"] = _clone_state_value(stochastic_seed)
                param_metadata["stochastic_seed"] = _stochastic_metadata(stochastic_seed)

            if param_metadata["quantized_states"] or "flash" in param_metadata:
                metadata["params"][param_id] = param_metadata

    state_dict[COMPRESSED_STATE_DICT_KEY] = metadata
    return state_dict


def _compressed_key(state_name: str) -> str:
    return f"{state_name}{QUANTIZED_SUFFIX}"


def _scales_key(state_name: str) -> str:
    return f"{state_name}{SCALES_SUFFIX}"


def _validate_param_groups(optimizer, state_dict):
    param_groups = state_dict.get("param_groups")
    if not isinstance(param_groups, list) or len(param_groups) != len(optimizer.param_groups):
        raise ValueError("Compressed checkpoint param_groups do not match the optimizer.")
    for group, saved_group in zip(optimizer.param_groups, param_groups):
        if len(group["params"]) != len(saved_group.get("params", [])):
            raise ValueError("Compressed checkpoint param group sizes do not match the optimizer.")


def _validate_spec_metadata(state_name: str, metadata: dict, scale_name: str, spec: QuantizedStateSpec):
    expected = {
        "scale_name": scale_name,
        "group_size": GROUP_SIZE,
        "signed": spec.signed,
        "sqrt": spec.sqrt,
        "softsign": spec.softsign,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Compressed checkpoint metadata mismatch for {state_name}: {key}.")


def _validate_compressed_state(optimizer, state_dict, specs):
    metadata = state_dict.get(COMPRESSED_STATE_DICT_KEY)
    if not isinstance(metadata, dict):
        raise ValueError("Compressed checkpoint metadata is missing.")
    if metadata.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported compressed checkpoint version.")
    if metadata.get("optimizer") != optimizer.__class__.__name__:
        raise ValueError("Compressed checkpoint optimizer class mismatch.")
    if not isinstance(metadata.get("params"), dict):
        raise ValueError("Compressed checkpoint parameter metadata is missing.")

    _validate_param_groups(optimizer, state_dict)
    saved_states = state_dict.get("state", {})
    param_metadata = metadata["params"]
    for group, saved_group in zip(optimizer.param_groups, state_dict["param_groups"]):
        for param, param_id in zip(group["params"], saved_group["params"]):
            saved_state = saved_states.get(param_id, {})
            metadata = param_metadata.get(param_id)
            has_compressed = any(key.endswith((QUANTIZED_SUFFIX, SCALES_SUFFIX)) for key in saved_state)
            if has_compressed and metadata is None:
                raise ValueError("Compressed checkpoint is missing required parameter metadata.")
            if metadata is None:
                continue
            if metadata.get("param_dtype") != _dtype_name(param.dtype):
                raise ValueError("Compressed checkpoint original parameter dtype mismatch.")

            quantized_metadata = metadata.get("quantized_states", {})
            for state_name, (scale_name, spec) in specs.items():
                qkey = _compressed_key(state_name)
                skey = _scales_key(state_name)
                if qkey not in saved_state and skey not in saved_state:
                    continue
                if qkey not in saved_state or skey not in saved_state or state_name not in quantized_metadata:
                    raise ValueError("Compressed checkpoint is missing required quantized state metadata.")
                _validate_spec_metadata(state_name, quantized_metadata[state_name], scale_name, spec)
                qstate = saved_state[qkey]
                scales = saved_state[skey]
                if quantized_metadata[state_name].get("dtype") != _dtype_name(spec.dtype):
                    raise ValueError("Compressed checkpoint quantized state spec mismatch.")
                if not torch.is_tensor(qstate) or qstate.dtype != spec.dtype or qstate.shape != param.shape:
                    raise ValueError("Compressed checkpoint quantized state spec mismatch.")
                if quantized_metadata[state_name].get("scale_dtype") != _dtype_name(torch.float32):
                    raise ValueError("Compressed checkpoint scale dtype mismatch.")
                if not torch.is_tensor(scales) or _dtype_name(scales.dtype) != quantized_metadata[state_name].get("scale_dtype"):
                    raise ValueError("Compressed checkpoint scale dtype mismatch.")
                if scales.shape != (quantized_scale_numel(param),):
                    raise ValueError("Compressed checkpoint group size mismatch.")

            flash_metadata = metadata.get("flash")
            flash_comp = saved_state.get("flash_comp")
            if flash_metadata is None:
                if torch.is_tensor(flash_comp) and flash_comp.dtype in (torch.int8, torch.int16):
                    raise ValueError("Compressed checkpoint is missing required FlashOptim metadata.")
                continue
            if flash_metadata.get("param_dtype") != _dtype_name(param.dtype):
                raise ValueError("Compressed checkpoint original parameter dtype mismatch.")
            if not torch.is_tensor(flash_comp):
                raise ValueError("Compressed checkpoint is missing flash_comp.")
            expected_dtype = torch.int16 if flash_metadata.get("precision") == "32" else torch.int8
            if flash_comp.shape != param.shape:
                raise ValueError("Compressed checkpoint flash_comp shape mismatch.")
            if (
                flash_metadata.get("precision") not in ("24", "32")
                or flash_metadata.get("dtype") != _dtype_name(expected_dtype)
                or flash_comp.dtype != expected_dtype
            ):
                raise ValueError("Compressed checkpoint flash precision mismatch.")

            stochastic_metadata = metadata.get("stochastic_seed")
            stochastic_seed = saved_state.get("stochastic_seed")
            if stochastic_metadata is None:
                if torch.is_tensor(stochastic_seed):
                    raise ValueError("Compressed checkpoint is missing required stochastic seed metadata.")
                continue
            if not torch.is_tensor(stochastic_seed) or stochastic_seed.dtype != torch.int32 or stochastic_seed.shape != torch.Size([]):
                raise ValueError("Compressed checkpoint stochastic seed metadata mismatch.")
            if stochastic_metadata.get("dtype") != _dtype_name(torch.int32) or tuple(stochastic_metadata.get("shape", ())) != ():
                raise ValueError("Compressed checkpoint stochastic seed metadata mismatch.")


def _loadable_public_state_dict(state_dict, specs):
    loadable = {
        "state": {},
        "param_groups": [dict(group) for group in state_dict["param_groups"]],
    }
    for param_id, state in state_dict["state"].items():
        public_state = {}
        for key, value in state.items():
            if key.endswith(SCALES_SUFFIX):
                continue
            if key.endswith(QUANTIZED_SUFFIX):
                state_name = key[: -len(QUANTIZED_SUFFIX)]
                _, spec = specs[state_name]
                scales = state[_scales_key(state_name)]
                public_state[state_name] = dequantize_state(value, scales, spec)
                continue
            public_state[key] = value
        loadable["state"][param_id] = public_state
    return loadable


def _restore_compressed_state_(optimizer, state_dict, specs):
    for group, saved_group in zip(optimizer.param_groups, state_dict["param_groups"]):
        for param, param_id in zip(group["params"], saved_group["params"]):
            saved_state = state_dict["state"].get(param_id, {})
            state = optimizer.state[param]
            for state_name, (scale_name, _spec) in specs.items():
                qkey = _compressed_key(state_name)
                skey = _scales_key(state_name)
                if qkey in saved_state and skey in saved_state:
                    state[state_name] = saved_state[qkey].to(device=param.device)
                    state[scale_name] = saved_state[skey].to(device=param.device)

            flash_comp = saved_state.get("flash_comp")
            if torch.is_tensor(flash_comp) and flash_comp.dtype in (torch.int8, torch.int16):
                state["flash_comp"] = flash_comp.to(device=param.device)

            stochastic_seed = saved_state.get("stochastic_seed")
            if torch.is_tensor(stochastic_seed) and stochastic_seed.dtype == torch.int32:
                state["stochastic_seed"] = stochastic_seed.to(device=param.device)


def load_compressed_state_dict(optimizer, state_dict):
    """Loads an optimi compressed checkpoint or a public optimizer state dict."""
    if COMPRESSED_STATE_DICT_KEY not in state_dict:
        optimizer.load_state_dict(state_dict)
        return

    specs = _optimizer_specs(optimizer)
    _validate_compressed_state(optimizer, state_dict, specs)
    optimizer.load_state_dict(_loadable_public_state_dict(state_dict, specs))
    _restore_compressed_state_(optimizer, state_dict, specs)
