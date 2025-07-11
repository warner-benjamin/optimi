---
title: ForEach Optimizers
---

# ForEach Optimizer Implementations

Like PyTorch, optimi supports foreach implementations of all optimizers. Foreach optimizers can be significantly faster than the for-loop versions.

Foreach implementations can increase optimizer peak memory usage. optimi attempts to reduce this extra overhead by reusing the gradient buffer for temporary variables. If the gradients are required between the optimization step and [gradient reset step](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad), set `foreach=False` to use the for-loop implementation.

??? note "Note: Foreach Requires PyTorch 2.1+"

    optimi’s foreach implementations require PyTorch 2.1 or newer.

If unspecified `foreach=None`, optimi will use the foreach implementation if training on a Cuda device.

## Example

Using a foreach implementation is as simple as setting `foreach=True` when initializing the optimizer.

```python
import torch
from torch import nn
from optimi import AdamW

# create model
model = nn.Linear(20, 1, device="cuda")

# initialize any optmi optimizer with `foreach=True`
opt = AdamW(model.parameters(), lr=1e-3, foreach=True)

# forward and backward
loss = model(torch.randn(20))
loss.backward()

# optimizer step is the foreach implementation
opt.step()
opt.zero_grad()
```