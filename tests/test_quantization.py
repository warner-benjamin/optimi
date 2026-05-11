import pytest
import torch

from optimi import AdamW, Lion, SGD
from optimi.quantization import ADAM_VARIANCE_STATE, MOMENTUM_STATE, dequantize_state, init_quantized_state, quantize_state_
from optimi.utils import HAS_TRITON


def _triton_device():
    if not HAS_TRITON:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return None


@pytest.mark.parametrize("spec", [MOMENTUM_STATE, ADAM_VARIANCE_STATE])
@pytest.mark.parametrize("n_elements", [0, 1, 31, 32, 33, 65])
def test_quantized_state_helpers_are_finite_for_tail_groups(spec, n_elements):
    values = torch.linspace(-2, 2, n_elements)
    if spec.sqrt:
        values = values.square()

    qstate, scales = init_quantized_state(values, spec)
    quantize_state_(values, qstate, scales, spec)
    restored = dequantize_state(qstate, scales, spec)

    assert qstate.dtype == spec.dtype
    assert scales.dtype == torch.float32
    assert scales.numel() == (n_elements + 31) // 32
    assert torch.isfinite(restored).all()
    assert restored.shape == values.shape


@pytest.mark.parametrize("optimizer_cls", [AdamW, Lion, SGD])
def test_quantize_requires_triton_true(optimizer_cls):
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="requires triton=True"):
        optimizer_cls([param], lr=1e-3, triton=False, quantize=True)


@pytest.mark.parametrize("optimizer_cls,kwargs", [(AdamW, {}), (Lion, {}), (SGD, {"momentum": 0.9}), (SGD, {"momentum": 0.0})])
def test_quantize_initializes_live_state(optimizer_cls, kwargs):
    device = _triton_device()
    if device is None:
        pytest.skip("requires a Triton-supported device")

    param = torch.nn.Parameter(torch.ones(65, device=device, dtype=torch.bfloat16))
    param.grad = torch.full_like(param, 0.25)
    optimizer = optimizer_cls([param], lr=1e-3, triton=True, flash=False, quantize=True, **kwargs)

    optimizer.step()

    state = optimizer.state[param]
    if optimizer_cls is AdamW:
        assert state["exp_avg"].dtype == torch.int8
        assert state["exp_avg_sq"].dtype == torch.uint8
        assert state["exp_avg_scales"].shape == (3,)
        assert state["exp_avg_sq_scales"].shape == (3,)
    elif optimizer_cls is SGD and kwargs["momentum"] == 0.0:
        assert "exp_avg_scales" not in state
    else:
        assert state["exp_avg"].dtype == torch.int8
        assert state["exp_avg_scales"].shape == (3,)


def test_quantize_requires_contiguous_params_and_grads():
    device = _triton_device()
    if device is None:
        pytest.skip("requires a Triton-supported device")

    base = torch.ones((2, 65), device=device, dtype=torch.bfloat16)
    param = torch.nn.Parameter(base.t())
    param.grad = torch.ones_like(param)
    optimizer = AdamW([param], lr=1e-3, triton=True, quantize=True)

    with pytest.raises(ValueError, match="requires contiguous parameters and gradients"):
        optimizer.step()

    assert optimizer.state[param] == {}


@pytest.mark.parametrize("optimizer_cls,kwargs", [(AdamW, {}), (Lion, {}), (SGD, {"momentum": 0.9})])
def test_quantized_state_dict_exports_dequantized_and_load_requantizes(optimizer_cls, kwargs):
    device = _triton_device()
    if device is None:
        pytest.skip("requires a Triton-supported device")

    param = torch.nn.Parameter(torch.randn(65, device=device, dtype=torch.bfloat16))
    param.grad = torch.randn_like(param)
    optimizer = optimizer_cls([param], lr=1e-3, triton=True, quantize=True, **kwargs)
    optimizer.step()
    live_state = optimizer.state[param]
    live_exp_avg = live_state["exp_avg"]
    live_exp_avg_scales = live_state["exp_avg_scales"]

    state_dict = optimizer.state_dict()
    saved_state = next(iter(state_dict["state"].values()))

    assert optimizer.state[param]["exp_avg"] is live_exp_avg
    assert optimizer.state[param]["exp_avg_scales"] is live_exp_avg_scales
    assert optimizer.state[param]["exp_avg"].dtype == torch.int8
    assert saved_state["exp_avg"].dtype == torch.float32
    assert "exp_avg_scales" not in saved_state
    if optimizer_cls is AdamW:
        assert saved_state["exp_avg_sq"].dtype == torch.float32
        assert "exp_avg_sq_scales" not in saved_state

    next_param = torch.nn.Parameter(torch.randn_like(param))
    next_optimizer = optimizer_cls([next_param], lr=1e-3, triton=True, quantize=True, **kwargs)
    next_optimizer.load_state_dict(state_dict)
    loaded_state = next_optimizer.state[next_param]

    assert loaded_state["exp_avg"].dtype == torch.int8
    assert loaded_state["exp_avg_scales"].dtype == torch.float32
    if optimizer_cls is AdamW:
        assert loaded_state["exp_avg_sq"].dtype == torch.uint8
        assert loaded_state["exp_avg_sq_scales"].dtype == torch.float32


@pytest.mark.parametrize("optimizer_cls,kwargs", [(AdamW, {}), (Lion, {}), (SGD, {"momentum": 0.9})])
def test_quantized_first_steps_track_unquantized(optimizer_cls, kwargs):
    device = _triton_device()
    if device is None:
        pytest.skip("requires a Triton-supported device")

    torch.manual_seed(1)
    initial = torch.randn(97, device=device, dtype=torch.bfloat16)
    grads = [torch.randn_like(initial) for _ in range(3)]
    param = torch.nn.Parameter(initial.clone())
    quantized_param = torch.nn.Parameter(initial.clone())
    optimizer = optimizer_cls([param], lr=1e-3, triton=True, flash=False, quantize=False, **kwargs)
    quantized_optimizer = optimizer_cls([quantized_param], lr=1e-3, triton=True, flash=False, quantize=True, **kwargs)

    for grad in grads:
        param.grad = grad.clone()
        quantized_param.grad = grad.clone()
        optimizer.step()
        quantized_optimizer.step()

    assert torch.allclose(param, quantized_param, atol=2e-2, rtol=0)
