---
title: Triton Optimizers
---

# Triton Optimizer Implementations

optimi's fused Triton optimizers are faster than PyTorch's fused Cuda optimizers, and nearly as fast as compiled optimizers without any hassle[^1].

![](https://ghp-cdn.benjaminwarner.dev/optimi/adamw_speed.png)

If unspecified `triton=None` and `foreach=None`, optimi will use the Triton implementation by default if training on a modern NVIDIA, AMD, or Intel GPU[^2].

??? note "Note: Triton Optimizers Requires PyTorch 2.6+"

    optimi’s Triton implementations require PyTorch 2.6 or newer. It's recommended to use the latest version of PyTorch and Triton.

The Triton backend is compatible with [gradient release](gradient_release.md) and [optimizer accumulation](optimizer_accumulation.md).

## Example

Using a Triton implementation is as simple as optionally setting `triton=True` when initializing the optimizer.

```python
import torch
from torch import nn
from optimi import AdamW

# create model
model = nn.Linear(20, 1, device="cuda")

# initialize any optmi optimizer with `triton=True`
# models on a supported GPU will default to `triton=True`
opt = AdamW(model.parameters(), lr=1e-3, triton=True)

# forward and backward
loss = model(torch.randn(20))
loss.backward()

# optimizer step is the Triton implementation
opt.step()
opt.zero_grad()
```

[^1]: Compiling optimizers requires [change to the training loop](https://docs.pytorch.org/tutorials/recipes/compiling_optimizer.html#setting-up-and-running-the-optimizer-benchmark) which might not be supported by your training framework of choice, and any dynamic hyperparemters such as the learning rate need to be [passed as Tensors or the optimizer will recompile every step](https://docs.pytorch.org/tutorials/recipes/compiling_optimizer_lr_scheduler.html).

[^2]: A GPU supporting bfloat16. Ampere or newer (A100 or RTX 3000 series), or any supported AMD or Intel GPU.