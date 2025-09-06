import pytest
import torch
from torch import nn

from optimi.utils import to_low_precision, MIN_TORCH_2_1

class RoPEBuffers(nn.Module):
    def __init__(self):
        super().__init__()
        # RoPE-like buffers that should remain FP32 by default
        self.register_buffer("cos", torch.ones(4, dtype=torch.float32))
        self.register_buffer("sin", torch.ones(4, dtype=torch.float32))


class FakeAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # Qualified name will include "rope" via this attribute
        self.rope = RoPEBuffers()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_proj(self.k_proj(self.q_proj(x)))


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 32):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.attn = FakeAttention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)
        # A non-RoPE buffer that should be cast to low precision by default
        self.register_buffer("global_scale", torch.ones(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(x)
        x = self.attn(x)
        x = self.norm(x)
        return self.head(x)


@pytest.mark.cpu
@pytest.mark.skipif(not MIN_TORCH_2_1, reason="requires PyTorch 2.1+")
def test_to_low_precision_default_preserves_embedding_and_rope_buffers():
    model = TinyLM()
    to_low_precision(model, dtype=torch.bfloat16)

    # Linear layer params should be low precision
    assert model.head.weight.dtype == torch.bfloat16

    # Embedding params should remain FP32 by default
    assert model.tok_embeddings.weight.dtype == torch.float32

    # RoPE buffers should remain FP32 by default
    assert model.attn.rope.cos.dtype == torch.float32
    assert model.attn.rope.sin.dtype == torch.float32

    # Non-RoPE buffer should be low precision
    assert model.global_scale.dtype == torch.bfloat16


@pytest.mark.cpu
@pytest.mark.skipif(not MIN_TORCH_2_1, reason="requires PyTorch 2.1+")
def test_to_low_precision_custom_fp32_modules_only_layernorm():
    model = TinyLM()
    # Only keep LayerNorm in FP32; embeddings should be cast to low precision now
    to_low_precision(model, dtype=torch.bfloat16, fp32_modules=(nn.LayerNorm,))

    assert model.norm.weight.dtype == torch.float32
    assert model.norm.bias.dtype == torch.float32

    # Embeddings should no longer be preserved as FP32 when overridden
    assert model.tok_embeddings.weight.dtype == torch.bfloat16

    # RoPE buffers still FP32 because keywords still match by default
    assert model.attn.rope.cos.dtype == torch.float32


@pytest.mark.cpu
@pytest.mark.skipif(not MIN_TORCH_2_1, reason="requires PyTorch 2.1+")
def test_to_low_precision_disable_rope_keywords_casts_rope_buffers():
    model = TinyLM()
    # Disable RoPE keyword preservation; buffers should be cast
    to_low_precision(model, dtype=torch.bfloat16, fp32_buffers=())

    assert model.attn.rope.cos.dtype == torch.bfloat16
    assert model.attn.rope.sin.dtype == torch.bfloat16

    # Embeddings remain FP32 by default
    assert model.tok_embeddings.weight.dtype == torch.float32


@pytest.mark.cpu
@pytest.mark.skipif(not MIN_TORCH_2_1, reason="requires PyTorch 2.1+")
def test_to_low_precision_fp32_buffers_accepts_module_types():
    model = TinyLM()

    # First, disable defaults and cast everything to low precision
    to_low_precision(model, dtype=torch.bfloat16, fp32_buffers=())
    assert model.attn.rope.cos.dtype == torch.bfloat16
    assert model.attn.rope.sin.dtype == torch.bfloat16

    # Now, re-run with module type selector to force RoPE buffers back to FP32
    to_low_precision(model, dtype=torch.bfloat16, fp32_buffers=(RoPEBuffers,))
    assert model.attn.rope.cos.dtype == torch.float32
    assert model.attn.rope.sin.dtype == torch.float32
