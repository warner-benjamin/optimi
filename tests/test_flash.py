import pytest
import torch

from optimi.flash import FlashPrecision, compute_flash_state, init_flash_state, normalize_flash_precision, reconstruct_fp32_param


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (False, FlashPrecision.OFF),
        ("off", FlashPrecision.OFF),
        (FlashPrecision.OFF, FlashPrecision.OFF),
        (True, FlashPrecision.EFFECTIVE_24),
        ("24", FlashPrecision.EFFECTIVE_24),
        (FlashPrecision.EFFECTIVE_24, FlashPrecision.EFFECTIVE_24),
        ("32", FlashPrecision.EFFECTIVE_32),
        (FlashPrecision.EFFECTIVE_32, FlashPrecision.EFFECTIVE_32),
    ],
)
def test_normalize_flash_precision(value, expected):
    assert normalize_flash_precision(value) == expected


def test_normalize_flash_precision_rejects_invalid_value():
    with pytest.raises(ValueError, match="Invalid flash precision"):
        normalize_flash_precision("16")


@pytest.mark.parametrize(
    ("flash", "expected_dtype"),
    [
        (None, None),
        (False, None),
        (FlashPrecision.OFF, None),
        (True, torch.int8),
        ("24", torch.int8),
        ("32", torch.int16),
    ],
)
def test_init_flash_state(flash, expected_dtype):
    param = torch.ones(3, 2, dtype=torch.bfloat16)
    state = init_flash_state(param, flash)
    if expected_dtype is None:
        assert state is None
    else:
        assert state is not None
        assert state.shape == param.shape
        assert state.dtype == expected_dtype
        assert state.device == param.device
        assert torch.count_nonzero(state) == 0


def test_init_flash_state_rejects_unsupported_dtype():
    param = torch.ones(3, dtype=torch.float32)
    with pytest.raises(ValueError, match="requires bf16 or fp16"):
        init_flash_state(param, True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("flash", [FlashPrecision.EFFECTIVE_24, FlashPrecision.EFFECTIVE_32])
def test_flash_roundtrip_error_bounds(dtype, flash):
    if dtype == torch.bfloat16:
        values = torch.tensor(
            [0.0, -0.0, 2.0**-133, -(2.0**-133), 2.0**-126, -(2.0**-126), 1.0, -1.0, 17.25, -31.5],
            dtype=torch.float32,
        )
        mantissa_bits = 7
        exponent_bias = 127
    else:
        values = torch.tensor(
            [0.0, -0.0, 2.0**-24, -(2.0**-24), 2.0**-14, -(2.0**-14), 1.0, -1.0, 17.25, -31.5],
            dtype=torch.float32,
        )
        mantissa_bits = 10
        exponent_bias = 15

    perturb = torch.linspace(-0.49, 0.49, values.numel(), dtype=torch.float32)
    narrow_base = values.to(dtype)
    log_ulp = _log_ulp(narrow_base, mantissa_bits, exponent_bias)
    fp32_param = narrow_base.to(torch.float32) + perturb * torch.exp2(log_ulp.to(torch.float32))
    narrow_param = fp32_param.to(dtype)

    flash_state = compute_flash_state(fp32_param, narrow_param, flash)
    reconstructed = reconstruct_fp32_param(narrow_param, flash_state)

    no_flash_error = (narrow_param.to(torch.float32) - fp32_param).abs()
    flash_error = (reconstructed - fp32_param).abs()
    assert torch.all(flash_error <= no_flash_error + torch.finfo(torch.float32).eps)

    signed_max = 127 if flash == FlashPrecision.EFFECTIVE_24 else 32767
    scale = torch.exp2((_log_ulp(narrow_param, mantissa_bits, exponent_bias) - 1).to(torch.float32))
    assert torch.all(flash_error <= scale / (2 * signed_max) + torch.finfo(torch.float32).eps)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("flash", [FlashPrecision.EFFECTIVE_24, FlashPrecision.EFFECTIVE_32])
def test_flash_exact_downcast_stores_zero(dtype, flash):
    fp32_param = torch.tensor([0.0, 1.0, -2.0, 16.0], dtype=torch.float32)
    narrow_param = fp32_param.to(dtype)

    flash_state = compute_flash_state(fp32_param, narrow_param, flash)
    reconstructed = reconstruct_fp32_param(narrow_param, flash_state)

    assert torch.count_nonzero(flash_state) == 0
    torch.testing.assert_close(reconstructed, fp32_param, rtol=0, atol=0)


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.float64, torch.int32])
def test_reconstruct_fp32_param_rejects_unsupported_param_dtype(bad_dtype):
    param = torch.ones(2, dtype=bad_dtype)
    flash_state = torch.zeros(2, dtype=torch.int8)
    with pytest.raises(ValueError, match="requires bf16 or fp16"):
        reconstruct_fp32_param(param, flash_state)


def test_reconstruct_fp32_param_rejects_unsupported_flash_state_dtype():
    param = torch.ones(2, dtype=torch.bfloat16)
    flash_state = torch.zeros(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="must be int8 or int16"):
        reconstruct_fp32_param(param, flash_state)


def test_compute_flash_state_rejects_unsupported_inputs():
    fp32_param = torch.ones(2, dtype=torch.float16)
    narrow_param = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must be float32"):
        compute_flash_state(fp32_param, narrow_param, True)

    fp32_param = torch.ones(2, dtype=torch.float32)
    narrow_param = torch.ones(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="requires bf16 or fp16"):
        compute_flash_state(fp32_param, narrow_param, True)


def test_compute_flash_state_stochastic_rounding_is_statistical():
    torch.manual_seed(1)
    narrow_param = torch.ones(4096, dtype=torch.bfloat16)
    scale = torch.full_like(narrow_param, 2.0**-8, dtype=torch.float32)
    fp32_param = narrow_param.to(torch.float32) + 0.25 * scale

    flash_state = compute_flash_state(fp32_param, narrow_param, True, stochastic_rounding=True)

    assert flash_state.dtype == torch.int8
    assert abs(flash_state.to(torch.float32).mean().item() - 31.75) < 0.75


def _log_ulp(param: torch.Tensor, mantissa_bits: int, exponent_bias: int) -> torch.Tensor:
    abs_bits = param.view(torch.int16).to(torch.int32).bitwise_and(0x7FFF)
    exponent_bits = abs_bits.bitwise_right_shift(mantissa_bits)
    return torch.where(
        exponent_bits == 0,
        torch.full_like(exponent_bits, 1 - exponent_bias - mantissa_bits),
        exponent_bits - exponent_bias - mantissa_bits,
    )
