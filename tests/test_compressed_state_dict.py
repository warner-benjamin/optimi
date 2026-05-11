import optimi
import pytest
import torch
from optimi import SGD, Adam, AdamW, Lion, StableAdamW
from optimi.quantization import COMPRESSED_STATE_DICT_KEY
from optimi.utils import HAS_TRITON


def _triton_device():
    if not HAS_TRITON:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return None


def _stepped_optimizer(optimizer_cls, kwargs):
    device = _triton_device()
    if device is None:
        pytest.skip("requires a Triton-supported device")

    param = torch.nn.Parameter(torch.ones(65, device=device, dtype=torch.bfloat16))
    param.grad = torch.full_like(param, 0.25)
    optimizer = optimizer_cls([param], lr=1e-3, flash=True, quantize=True, triton=True, **kwargs)
    optimizer.step()
    return optimizer, param


@pytest.mark.parametrize(
    "optimizer_cls,kwargs,state_names",
    [
        (Adam, {"decouple_wd": True}, ("exp_avg", "exp_avg_sq")),
        (AdamW, {}, ("exp_avg", "exp_avg_sq")),
        (Lion, {}, ("exp_avg",)),
        (SGD, {"momentum": 0.9}, ("exp_avg",)),
    ],
)
def test_compressed_state_dict_roundtrip_preserves_live_integer_state(optimizer_cls, kwargs, state_names):
    """Compressed checkpoints preserve live quantized tensors, scales, and flash_comp."""
    optimizer, param = _stepped_optimizer(optimizer_cls, kwargs)
    checkpoint = optimi.compressed_state_dict(optimizer)
    saved_state = next(iter(checkpoint["state"].values()))

    assert COMPRESSED_STATE_DICT_KEY in checkpoint
    assert checkpoint[COMPRESSED_STATE_DICT_KEY]["optimizer"] == optimizer_cls.__name__
    assert saved_state["flash_comp"].dtype == torch.int8
    for state_name in state_names:
        assert f"{state_name}::quantized" in saved_state
        assert f"{state_name}::scales" in saved_state
        assert state_name not in saved_state

    next_param = torch.nn.Parameter(torch.ones_like(param))
    next_optimizer = optimizer_cls([next_param], lr=1e-3, flash=True, quantize=True, triton=True, **kwargs)
    optimi.load_compressed_state_dict(next_optimizer, checkpoint)

    state = optimizer.state[param]
    next_state = next_optimizer.state[next_param]
    assert torch.equal(next_state["flash_comp"], state["flash_comp"])
    for state_name in state_names:
        scale_name = f"{state_name}_scales"
        if optimizer_cls is AdamW and state_name == "exp_avg_sq":
            scale_name = "exp_avg_sq_scales"
        assert next_state[state_name].dtype == state[state_name].dtype
        assert torch.equal(next_state[state_name], state[state_name])
        assert torch.equal(next_state[scale_name], state[scale_name])


def test_regular_load_state_dict_rejects_compressed_checkpoint():
    """Regular optimizer load_state_dict rejects optimi compressed checkpoints."""
    optimizer, param = _stepped_optimizer(AdamW, {})
    checkpoint = optimi.compressed_state_dict(optimizer)
    next_optimizer = AdamW([torch.nn.Parameter(torch.ones_like(param))], lr=1e-3, flash=True, quantize=True, triton=True)

    with pytest.raises(ValueError, match="load_compressed_state_dict"):
        next_optimizer.load_state_dict(checkpoint)


def test_base_load_state_dict_rejects_compressed_checkpoint_for_other_optimizers():
    """Base optimizer load_state_dict rejects compressed checkpoints format-wide."""
    optimizer, param = _stepped_optimizer(AdamW, {})
    checkpoint = optimi.compressed_state_dict(optimizer)
    next_optimizer = StableAdamW([torch.nn.Parameter(torch.ones_like(param))], lr=1e-3, triton=True)

    with pytest.raises(ValueError, match="load_compressed_state_dict"):
        next_optimizer.load_state_dict(checkpoint)


def test_load_compressed_state_dict_accepts_public_state_dict():
    """Compressed loader accepts regular public state dicts as a convenience."""
    optimizer, param = _stepped_optimizer(AdamW, {})
    checkpoint = optimizer.state_dict()
    next_optimizer = AdamW([torch.nn.Parameter(torch.ones_like(param))], lr=1e-3, flash=True, quantize=True, triton=True)

    optimi.load_compressed_state_dict(next_optimizer, checkpoint)

    state = next(iter(next_optimizer.state.values()))
    assert state["exp_avg"].dtype == torch.int8
    assert state["exp_avg_sq"].dtype == torch.uint8
    assert "exp_avg_scales" in state
    assert "exp_avg_sq_scales" in state


def test_compressed_state_dict_preserves_stochastic_seed():
    """Compressed checkpoints directly restore stochastic rounding stream state."""
    optimizer, param = _stepped_optimizer(AdamW, {"stochastic_rounding": True})
    checkpoint = optimi.compressed_state_dict(optimizer)
    next_param = torch.nn.Parameter(torch.ones_like(param))
    next_optimizer = AdamW(
        [next_param],
        lr=1e-3,
        flash=True,
        quantize=True,
        stochastic_rounding=True,
        triton=True,
    )

    optimi.load_compressed_state_dict(next_optimizer, checkpoint)

    assert next_optimizer.state[next_param]["stochastic_seed"].dtype == torch.int32
    assert torch.equal(next_optimizer.state[next_param]["stochastic_seed"], optimizer.state[param]["stochastic_seed"])


def _first_param_metadata(checkpoint):
    metadata = checkpoint[COMPRESSED_STATE_DICT_KEY]
    param_id = next(iter(metadata["params"]))
    return param_id, metadata["params"][param_id]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda checkpoint: checkpoint[COMPRESSED_STATE_DICT_KEY].update(version=0), "version"),
        (lambda checkpoint: checkpoint[COMPRESSED_STATE_DICT_KEY].update(optimizer="Lion"), "optimizer class"),
        (lambda checkpoint: _first_param_metadata(checkpoint)[1].update(param_dtype="torch.float32"), "parameter dtype"),
        (
            lambda checkpoint: _first_param_metadata(checkpoint)[1]["quantized_states"]["exp_avg"].update(group_size=16),
            "group_size",
        ),
        (
            lambda checkpoint: _first_param_metadata(checkpoint)[1]["quantized_states"]["exp_avg"].update(signed=False),
            "metadata mismatch",
        ),
        (
            lambda checkpoint: next(iter(checkpoint["state"].values())).__setitem__(
                "exp_avg::scales", next(iter(checkpoint["state"].values()))["exp_avg::scales"].half()
            ),
            "scale dtype",
        ),
        (
            lambda checkpoint: next(iter(checkpoint["state"].values())).__setitem__(
                "exp_avg::quantized", next(iter(checkpoint["state"].values()))["exp_avg::quantized"].reshape(5, 13)
            ),
            "quantized state spec",
        ),
        (
            lambda checkpoint: next(iter(checkpoint["state"].values())).__setitem__(
                "flash_comp", next(iter(checkpoint["state"].values()))["flash_comp"].reshape(5, 13)
            ),
            "flash_comp shape",
        ),
        (lambda checkpoint: _first_param_metadata(checkpoint)[1]["flash"].update(precision="32"), "flash precision"),
        (lambda checkpoint: checkpoint[COMPRESSED_STATE_DICT_KEY]["params"].clear(), "required parameter metadata"),
        (
            lambda checkpoint: _first_param_metadata(checkpoint)[1]["quantized_states"].pop("exp_avg"),
            "required quantized state metadata",
        ),
    ],
)
def test_load_compressed_state_dict_validates_metadata_before_loading(mutate, match):
    """Compressed loader validates checkpoint metadata before mutating optimizer state."""
    optimizer, param = _stepped_optimizer(AdamW, {})
    checkpoint = optimi.compressed_state_dict(optimizer)
    mutate(checkpoint)
    next_param = torch.nn.Parameter(torch.ones_like(param))
    next_optimizer = AdamW([next_param], lr=1e-3, flash=True, quantize=True, triton=True)

    with pytest.raises(ValueError, match=match):
        optimi.load_compressed_state_dict(next_optimizer, checkpoint)

    assert next_optimizer.state[next_param] == {}


def test_load_compressed_state_dict_validates_stochastic_seed_metadata():
    """Compressed loader rejects malformed stochastic seed state before loading."""
    optimizer, param = _stepped_optimizer(AdamW, {"stochastic_rounding": True})
    checkpoint = optimi.compressed_state_dict(optimizer)
    next(iter(checkpoint["state"].values()))["stochastic_seed"] = torch.ones(1, device=param.device, dtype=torch.int32)
    next_param = torch.nn.Parameter(torch.ones_like(param))
    next_optimizer = AdamW(
        [next_param],
        lr=1e-3,
        flash=True,
        quantize=True,
        stochastic_rounding=True,
        triton=True,
    )

    with pytest.raises(ValueError, match="stochastic seed"):
        optimi.load_compressed_state_dict(next_optimizer, checkpoint)

    assert next_optimizer.state[next_param] == {}
