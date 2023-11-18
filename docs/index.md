---
title: "optimi"
description: "Fast, Modern, and Low Precision PyTorch Optimizers"
---

# optimī

**Fast, Modern, and Low Precision PyTorch Optimizers**

optimi enables accurate low precision training via Kahan summation, supports fully decoupled weight decay, and features fast implementations of modern optimizers.

## Low Precision Training with Kahan Summation

optimi optimizers can match the performance of mixed precision when [training in BFloat16 by using Kahan summation](kahan_summation.md).

Training in BFloat16 with Kahan summation can reduce non-activation training memory usage by [37.5 to 45.5 percent](kahan_summation.md/#memory-savings) when using an Adam optimizer. BFloat16 training increases single GPU [training speed by ~10 percent](kahan_summation.md/#training-speedup) at the same batch size.

## Fully Decoupled Weight Decay

In addition to supporting PyTorch-style decoupled weight decay, optimi optimizers also supports [fully decoupled weight decay](fully_decoupled_weight_decay.md).

Fully decoupled weight decay decouples weight decay from the learning rate, more accurately following [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101). This can help simplify hyperparameter tuning as the optimal weight decay is no longer tied to the learning rate.

## Foreach Implementations

All optimi optimizers have fast [foreach implementations](foreach.md), which can significantly outperform the for-loop versions. optimi reuses the gradient buffer for temporary variables to reduce foreach memory usage.

## Install

optimi is available to install from pypi.

```bash
pip install optimi
```

## Usage

To use an optimi optimizer with Kahan summation and fully decoupled weight decay:

```python
import torch
from torch import nn
from optimi import AdamW

# create or cast model in low precision (bfloat16)
model = nn.Linear(20, 1, dtype=torch.bfloat16)

# instantiate AdamW with parameters and fully decoupled weight decay
# Kahan summation is automatically enabled since model & inputs are bfloat16
opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5, decouple_lr=True)

# forward and backward, casting input to bfloat16 if needed
loss = model(torch.randn(20, dtype=torch.bfloat16))
loss.backward()

# optimizer step
opt.step()
opt.zero_grad()
```

To use with PyTorch-style weight decay with float32 or mixed precision:

```python
# create model
model = nn.Linear(20, 1)

# instantiate AdamW with parameters
opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

## Difference from PyTorch

optimi optimizers do not support compilation, differentiation, or have capturable versions.

optimi Adam optimizers do not support AMSGrad and SGD does not support Nesterov momentum. Optimizers which debias updates (Adam optimizers and Adan) calculate the debias term per parameter group, not per parameter.

## Optimizers

optimi implements the following optimizers:

* [Adam](optimizers/adam.md)
* [AdamW](optimizers/adamw.md)
* [Adan](optimizers/adan.md)
* [Lion](optimizers/lion.md)
* [SGD](optimizers/sgd.md)
* [StableAdamW](optimizers/stableadamw.md)