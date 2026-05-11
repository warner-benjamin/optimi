import pytest
import torch
from optimi import Adam, AdamW, FlashPrecision
from optimi.adam import adam
from optimi.flash import init_flash_state
from optimi.utils import HAS_TRITON


def _triton_device():
    if not HAS_TRITON:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return None


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
def test_adam_flash_requires_triton_true(optimizer_cls):
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="requires triton=True"):
        optimizer_cls([param], lr=1e-3, flash=True, triton=False)


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
def test_adam_flash_rejects_kahan_sum_conflict(optimizer_cls):
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="kahan_sum=True and flash cannot be used together"):
        optimizer_cls([param], lr=1e-3, kahan_sum=True, flash=True, triton=True)


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
@pytest.mark.parametrize("kwargs", [{"quantize": True}, {"stochastic_rounding": True}])
def test_adam_flash_reserved_options_raise(optimizer_cls, kwargs):
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="not supported until"):
        optimizer_cls([param], lr=1e-3, triton=True, **kwargs)


def test_adam_explicit_flash_rejects_cpu_before_state_init():
    if not HAS_TRITON:
        pytest.skip("requires Triton import support")
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    param.grad = torch.ones_like(param)
    optimizer = Adam([param], lr=1e-3, flash=True, triton=True)

    with pytest.raises(ValueError, match="requires Triton-supported GPU/XPU"):
        optimizer.step()

    assert optimizer.state[param] == {}


@pytest.mark.parametrize("precision", [FlashPrecision.EFFECTIVE_24, FlashPrecision.EFFECTIVE_32])
def test_adam_flash_comp_uses_default_state_dict_load(precision):
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    optimizer = Adam([param], lr=1e-3, triton=False)
    flash_comp = init_flash_state(param, precision)
    optimizer.state[param]["flash_comp"] = flash_comp

    state_dict = optimizer.state_dict()
    next_optimizer = Adam([torch.nn.Parameter(torch.ones_like(param))], lr=1e-3, triton=False)
    next_optimizer.load_state_dict(state_dict)

    assert "flash_comp" in next(iter(state_dict["state"].values()))
    assert torch.equal(next(iter(next_optimizer.state.values()))["flash_comp"], flash_comp)


def test_adam_disabled_flash_ignores_stale_flash_comp_state():
    param = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    param.grad = torch.ones_like(param)
    optimizer = Adam([param], lr=1e-3, flash=False, triton=False)
    stale_flash_comp = init_flash_state(param, True)
    optimizer.state[param]["flash_comp"] = stale_flash_comp.clone()

    optimizer.step()

    assert torch.equal(optimizer.state[param]["flash_comp"], stale_flash_comp)


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
def test_adam_flash_triton_step_initializes_flash_comp(optimizer_cls):
    device = _triton_device()
    if device is None:
        pytest.skip("requires a Triton-supported device")
    param = torch.nn.Parameter(torch.ones(32, device=device, dtype=torch.float16))
    param.grad = torch.full_like(param, 0.25)
    optimizer = optimizer_cls([param], lr=1e-3, flash=True, triton=True)

    optimizer.step()

    assert optimizer.state[param]["flash_comp"].dtype == torch.int8
    assert torch.isfinite(param).all()


def test_adam_functional_flash_rejects_non_triton():
    param = torch.ones(2, dtype=torch.bfloat16)
    grad = torch.ones_like(param)
    exp_avg = torch.zeros_like(param)
    exp_avg_sq = torch.zeros_like(param)
    step = torch.tensor(0, dtype=torch.int32)
    flash_comp = init_flash_state(param, True)

    with pytest.raises(ValueError, match="requires triton=True"):
        adam(
            params=[param],
            grads=[grad],
            exp_avgs=[exp_avg],
            exp_avg_sqs=[exp_avg_sq],
            flash_comps=[flash_comp],
            lr=1e-3,
            beta1=0.9,
            beta2=0.99,
            weight_decay=0.0,
            eps=1e-6,
            step=step,
            decouple_wd=True,
            triton=False,
        )


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
def test_adam_flash_gradient_release_step_initializes_flash_comp(optimizer_cls):
    device = _triton_device()
    if device is None:
        pytest.skip("requires a Triton-supported device")
    param = torch.nn.Parameter(torch.ones(32, device=device, dtype=torch.float16))
    param.grad = torch.full_like(param, 0.25)
    optimizer = optimizer_cls([param], lr=1e-3, flash=True, triton=True, gradient_release=True)

    optimizer.step(param=param)

    assert optimizer.state[param]["flash_comp"].dtype == torch.int8
    assert optimizer.state[param]["flash_comp"].abs().sum() > 0
    assert torch.isfinite(param).all()


def test_adam_flash_gradient_release_accumulation_skips_param_and_flash_update():
    device = _triton_device()
    if device is None:
        pytest.skip("requires a Triton-supported device")
    param = torch.nn.Parameter(torch.ones(32, device=device, dtype=torch.float16))
    param.grad = torch.full_like(param, 0.25)
    optimizer = Adam([param], lr=1e-3, flash=True, triton=True, gradient_release=True)
    optimizer.step(param=param)
    before_param = param.detach().clone()
    before_flash = optimizer.state[param]["flash_comp"].clone()
    before_exp_avg = optimizer.state[param]["exp_avg"].clone()

    param.grad = torch.full_like(param, 0.5)
    optimizer.optimizer_accumulation = True
    optimizer.step(param=param)

    assert torch.equal(param, before_param)
    assert torch.equal(optimizer.state[param]["flash_comp"], before_flash)
    assert not torch.equal(optimizer.state[param]["exp_avg"], before_exp_avg)
