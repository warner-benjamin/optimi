# Copyright (c) 2023-present Benjamin Warner
# SPDX-License-Identifier: MIT

from enum import Enum

import torch
from torch import Tensor

from optimi.utils import HAS_TRITON

__all__ = [
    "FlashPrecision",
    "compute_flash_state",
    "effective_flash_precision",
    "init_flash_state",
    "normalize_flash_precision",
    "reconstruct_fp32_param",
    "resolve_effective_flash",
    "uses_flash",
]


class FlashPrecision(str, Enum):
    """Effective FlashOptim parameter precision."""

    OFF = "off"
    EFFECTIVE_24 = "24"
    EFFECTIVE_32 = "32"


_MANTISSA_BITS = {
    torch.float16: 10,
    torch.bfloat16: 7,
}
_EXPONENT_BIAS = {
    torch.float16: 15,
    torch.bfloat16: 127,
}
_FLASH_STATE_DTYPES = {
    FlashPrecision.EFFECTIVE_24: torch.int8,
    FlashPrecision.EFFECTIVE_32: torch.int16,
}
_SIGNED_MAX = {
    torch.int8: 127,
    torch.int16: 32767,
}


def normalize_flash_precision(flash: bool | str | FlashPrecision | None) -> FlashPrecision | None:
    """Normalize public flash precision input to the internal enum."""
    if flash is None:
        return None
    if flash is True:
        return FlashPrecision.EFFECTIVE_24
    if flash is False:
        return FlashPrecision.OFF
    if isinstance(flash, FlashPrecision):
        return flash
    if isinstance(flash, str):
        try:
            return FlashPrecision(flash)
        except ValueError:
            pass
    raise ValueError(f"Invalid flash precision: {flash=}")


def _flash_state_dtype(flash: bool | str | FlashPrecision | None) -> torch.dtype | None:
    precision = normalize_flash_precision(flash)
    if precision in (None, FlashPrecision.OFF):
        return None
    return _FLASH_STATE_DTYPES[precision]


def uses_flash(flash: bool | str | FlashPrecision | None) -> bool:
    """Return whether a requested flash precision explicitly enables compensation."""
    return normalize_flash_precision(flash) not in (None, FlashPrecision.OFF)


def effective_flash_precision(param: Tensor, flash: bool | str | FlashPrecision | None, triton: bool) -> FlashPrecision | None:
    """Resolve per-parameter FlashOptim precision after execution-path selection."""
    if param.dtype not in _MANTISSA_BITS:
        return None
    precision = normalize_flash_precision(flash)
    if precision is None:
        return FlashPrecision.EFFECTIVE_24 if triton else None
    if precision == FlashPrecision.OFF:
        return None
    return precision


def resolve_effective_flash(
    params: list[Tensor],
    grads: list[Tensor],
    flash: bool | str | FlashPrecision | None,
    triton: bool,
    triton_supported: bool,
) -> list[FlashPrecision | None]:
    """Resolve group FlashOptim precision and validate explicit FlashOptim requirements."""
    explicit_flash = uses_flash(flash)
    if explicit_flash and not triton_supported:
        raise ValueError("FlashOptim training support requires Triton-supported GPU/XPU parameters.")

    effective_flashes = [effective_flash_precision(param, flash, triton) for param in params]
    if explicit_flash:
        if not any(effective_flash is not None for effective_flash in effective_flashes):
            raise ValueError("flash was enabled, but no bf16 or fp16 parameters were found.")
        for param, grad, effective_flash in zip(params, grads, effective_flashes):
            if effective_flash is not None and (not param.is_contiguous() or not grad.is_contiguous()):
                raise ValueError("FlashOptim training support requires contiguous parameters and gradients.")

    return effective_flashes


def _check_narrow_param(param: Tensor):
    if param.dtype not in _MANTISSA_BITS:
        raise ValueError(f"FlashOptim parameter compensation requires bf16 or fp16 parameters, got {param.dtype}.")


def init_flash_state(param: Tensor, flash: bool | str | FlashPrecision | None) -> Tensor | None:
    """Initialize FlashOptim parameter compensation state for a resolved precision."""
    flash_dtype = _flash_state_dtype(flash)
    if flash_dtype is None:
        return None
    _check_narrow_param(param)
    return torch.zeros_like(param, dtype=flash_dtype, memory_format=torch.preserve_format)


def _log_ulp(param: Tensor) -> Tensor:
    mantissa_bits = _MANTISSA_BITS[param.dtype]
    exponent_bias = _EXPONENT_BIAS[param.dtype]
    abs_bits = param.view(torch.int16).to(torch.int32).bitwise_and(0x7FFF)
    exponent_bits = abs_bits.bitwise_right_shift(mantissa_bits)
    return torch.where(
        exponent_bits == 0,
        torch.full_like(exponent_bits, 1 - exponent_bias - mantissa_bits),
        exponent_bits - exponent_bias - mantissa_bits,
    )


def _exp2_scale(log_scale: Tensor) -> Tensor:
    return torch.exp2(log_scale.to(torch.float32))


def reconstruct_fp32_param(param: Tensor, flash_state: Tensor) -> Tensor:
    """Reconstruct an fp32 parameter from a narrow parameter and FlashOptim state."""
    _check_narrow_param(param)
    if flash_state.dtype not in _SIGNED_MAX:
        raise ValueError(f"flash_state must be int8 or int16, got {flash_state.dtype}.")
    if flash_state.shape != param.shape:
        raise ValueError(f"flash_state shape must match param shape, got {flash_state.shape=} and {param.shape=}.")

    scale = _exp2_scale(_log_ulp(param) - 1)
    correction = flash_state.to(torch.float32) / _SIGNED_MAX[flash_state.dtype] * scale
    return param.to(torch.float32) + correction


def compute_flash_state(
    fp32_param: Tensor,
    narrow_param: Tensor,
    flash: bool | str | FlashPrecision,
    stochastic_rounding: bool = False,
) -> Tensor:
    """Compute FlashOptim state from an fp32 parameter and its narrow storage tensor."""
    if stochastic_rounding:
        raise ValueError("stochastic_rounding=True is not supported until the stochastic rounding milestone.")
    if fp32_param.dtype != torch.float32:
        raise ValueError(f"fp32_param must be float32, got {fp32_param.dtype}.")
    _check_narrow_param(narrow_param)
    if fp32_param.shape != narrow_param.shape:
        raise ValueError(f"fp32_param shape must match narrow_param shape, got {fp32_param.shape=} and {narrow_param.shape=}.")

    flash_dtype = _flash_state_dtype(flash)
    if flash_dtype is None:
        raise ValueError(f"compute_flash_state requires enabled flash precision, got {flash=}.")

    scale = _exp2_scale(_log_ulp(narrow_param) - 1)
    error = fp32_param - narrow_param.to(torch.float32)
    normalized = (error / scale).clamp(-1.0, 1.0)
    scaled = normalized * _SIGNED_MAX[flash_dtype]
    rounded = torch.copysign(torch.floor(scaled.abs() + 0.5), scaled)
    return rounded.to(flash_dtype)


if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _flash_unbiased_exponent(param, mantissa_bits: tl.constexpr):
        param = tl.abs(param)
        if mantissa_bits == 7:
            param_bits = param.to(tl.uint16, bitcast=True)
            exponent_bias = 127
        elif mantissa_bits == 10:
            param_bits = param.to(tl.uint16, bitcast=True)
            exponent_bias = 15
        else:
            raise ValueError(f"Unsupported mantissa_bits: {mantissa_bits}")

        exponent_bits = param_bits >> mantissa_bits
        return tl.where(exponent_bits == 0, 1 - exponent_bias, exponent_bits.to(tl.int32) - exponent_bias)

    @triton.jit
    def _flash_log_scale(param, mantissa_bits: tl.constexpr):
        return _flash_unbiased_exponent(param, mantissa_bits) - mantissa_bits - 1

    @triton.jit
    def _scale_by_exp2(value, log_scale):
        log_scale = log_scale.to(tl.float32)
        half_scale = tl.floor(log_scale / 2.0)
        return value * tl.exp2(half_scale) * tl.exp2(log_scale - half_scale)

    @triton.jit
    def _apply_flash(param, flash_comp, mantissa_bits: tl.constexpr, signed_max: tl.constexpr):
        scale = _flash_log_scale(param, mantissa_bits)
        correction = _scale_by_exp2(flash_comp.to(tl.float32) / signed_max, scale)
        return param.to(tl.float32) + correction

    @triton.jit
    def _compute_flash(fp32_param, narrow_param, flash_t: tl.constexpr, mantissa_bits: tl.constexpr, signed_max: tl.constexpr):
        scale = _flash_log_scale(narrow_param, mantissa_bits)
        error = fp32_param - narrow_param.to(tl.float32)
        normalized = tl.clamp(_scale_by_exp2(error, -scale), -1.0, 1.0)
        scaled = normalized * signed_max
        rounded_abs = tl.floor(tl.abs(scaled) + 0.5)
        rounded = tl.where(scaled < 0, -rounded_abs, rounded_abs)
        return rounded.to(flash_t)
