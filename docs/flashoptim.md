---
title: FlashOptim Training
---

# FlashOptim Training

FlashOptim training stores low precision model parameters with a small integer correction tensor. During a Triton optimizer step, optimi reconstructs an effective higher precision parameter, applies the optimizer update in Float32, writes the low precision parameter, then recomputes the correction.

FlashOptim is available for `Adam`, `AdamW`, `Lion`, and `SGD`.

## Options

Pass `flash=True` or `flash="24"` to store an `int8` correction tensor for BFloat16 or Float16 parameters. Pass `flash="32"` to store an `int16` correction tensor.

```python
import torch
from optimi import AdamW

model = torch.nn.Linear(20, 1, device="cuda", dtype=torch.bfloat16)
opt = AdamW(model.parameters(), lr=1e-3, flash=True, triton=True)
```

When `flash=None`, optimi automatically enables FlashOptim parameter compensation for BFloat16 and Float16 parameters only when the resolved backend is Triton. Float32 parameters are left unchanged.

Explicit FlashOptim training options require `triton=True`:

- `flash=True`, `flash="24"`, or `flash="32"`
- `quantize=True`
- `stochastic_rounding=True`

## Quantized Optimizer State

`quantize=True` stores supported optimizer state with grouped 8-bit quantization:

- Adam and AdamW quantize `exp_avg` and `exp_avg_sq`.
- Lion quantizes `exp_avg`.
- SGD quantizes the momentum buffer when `momentum != 0`.

```python
opt = AdamW(model.parameters(), lr=1e-3, flash=True, quantize=True, triton=True)
```

Default `state_dict()` output stays PyTorch-compatible. Quantized live optimizer state is exported as dequantized floating tensors, and scale tensors are not exposed in default checkpoints.

Use `optimi.compressed_state_dict(optimizer)` and `optimi.load_compressed_state_dict(optimizer, state_dict)` to save and load optimi compressed checkpoints that preserve quantized integer state, scales, FlashOptim metadata, and `flash_comp` directly.

## Stochastic Rounding

`stochastic_rounding=True` applies stochastic rounding to FlashOptim correction bits after the low precision parameter has already been written. It does not stochastic-round every low precision tensor write.

```python
opt = AdamW(model.parameters(), lr=1e-3, flash=True, stochastic_rounding=True, triton=True)
```

Each parameter owns a deterministic stochastic-rounding stream initialized from PyTorch's RNG. Reusing the same `torch.manual_seed(...)`, model initialization, and optimizer construction order reproduces the same FlashOptim correction stream.

## Support Matrix

| Optimizer | `flash` | `quantize` | `stochastic_rounding` | Gradient release |
| --- | --- | --- | --- | --- |
| Adam | yes | yes | yes | yes |
| AdamW | yes | yes | yes | yes |
| Lion | yes | yes | yes | yes |
| SGD | yes | yes | yes | yes |
| Adan | no | no | no | no |
| RAdam | no | no | no | no |
| Ranger | no | no | no | no |
| StableAdamW | no | no | no | no |

Regular `load_state_dict()` rejects optimi compressed checkpoints. Load compressed checkpoints with `optimi.load_compressed_state_dict(...)`.

## Kahan Summation

FlashOptim is the replacement path for Kahan summation on supported Triton optimizers. `kahan_sum=True` and explicit `flash` cannot be used together.

Kahan summation remains available for non-Triton training and for optimizers that do not yet support FlashOptim. optimi does not emit a Kahan deprecation warning while Kahan remains the only low precision compensation path for some optimizers.
