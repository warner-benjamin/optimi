---
title: "optimi"
description: "Fast, Modern, Memory Efficient, and Low Precision PyTorch Optimizers"
---

# optimī

**Fast, Modern, Memory Efficient, and Low Precision PyTorch Optimizers**

optimi enables accurate low precision training via Kahan summation, integrates gradient release and optimizer accumulation for additional memory efficiency, supports fully decoupled weight decay, and features fast implementations of modern optimizers.

## Low Precision Training with Kahan Summation

optimi optimizers can nearly reach or match the performance of mixed precision when [training in BFloat16 by using Kahan summation](kahan_summation.md).

Training in BFloat16 with Kahan summation can reduce non-activation training memory usage by [37.5 to 45.5 percent](kahan_summation.md/#memory-savings) when using an Adam optimizer. BFloat16 training increases single GPU [training speed by ~10 percent](kahan_summation.md/#training-speedup) at the same batch size.

## Gradient Release: Fused Backward and Optimizer Step

optimi optimizers can perform the [optimization step layer-by-layer during the backward pass](gradient_release.md), immediately freeing gradient memory.

Unlike the current PyTorch implementation, optimi’s gradient release optimizers are a drop-in replacement for standard optimizers and seamlessly work with exisiting hyperparmeter schedulers.

## Optimizer Accumulation: Gradient Release and Accumulation

optimi optimizers can approximate gradient accumulation with gradient release by [accumulating gradients into the optimizer states](optimizer_accumulation.md).

## Fully Decoupled Weight Decay

In addition to supporting PyTorch-style decoupled weight decay, optimi optimizers also support [fully decoupled weight decay](fully_decoupled_weight_decay.md).

Fully decoupled weight decay decouples weight decay from the learning rate, more accurately following [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101). This can help simplify hyperparameter tuning as the optimal weight decay is no longer tied to the learning rate.

## Foreach Implementations

All optimi optimizers have fast [foreach implementations](foreach.md), which can significantly outperform the for-loop versions. optimi reuses the gradient buffer for temporary variables to reduce foreach memory usage.

## Install

optimi is available to install from pypi.

```bash
pip install torch-optimi
```

## Usage

To use an optimi optimizer with Kahan summation and fully decoupled weight decay:

```python
import torch
from torch import nn
from optimi import AdamW

# create or cast model in low precision (bfloat16)
model = nn.Linear(20, 1, dtype=torch.bfloat16)

# initialize any optimi optimizer with parameters & fully decoupled weight decay
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

# initialize any optimi optimizer with parameters
opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

To use with gradient release:

```python
# initialize any optimi optimizer with `gradient_release=True`
# and call `prepare_for_gradient_release` on model and optimizer
opt = AdamW(model.parameters(), lr=1e-3, gradient_release=True)
prepare_for_gradient_release(model, opt)

# calling backward on the model will peform the optimzier step
loss = model(torch.randn(20, dtype=torch.bfloat16))
loss.backward()

# optimizer step and zero_grad are no longer needed, and will
# harmlessly no-op if called by an existing training framework
# opt.step()
# opt.zero_grad()

# optionally remove gradient release hooks when done training
remove_gradient_release(model)
```

To use with optimizer accumulation:

```python
# initialize any optimi optimizer with `gradient_release=True`
# and call `prepare_for_gradient_release` on model and optimizer
opt = AdamW(model.parameters(), lr=1e-3, gradient_release=True)
prepare_for_gradient_release(model, opt)

# update model parameters every four steps after accumulating
# gradients directly into the optimizer states
accumulation_steps = 4

# use existing PyTorch dataloader
for idx, batch in enumerate(dataloader):
    # `optimizer_accumulation=True` accumulates gradients into
    # optimizer states. set `optimizer_accumulation=False` to
    # update parameters by performing a full gradient release step
    opt.optimizer_accumulation = (idx+1) % accumulation_steps != 0

    # calling backward on the model will peform the optimizer step
    # either accumulating gradients or updating model parameters
    loss = model(batch)
    loss.backward()

    # optimizer step and zero_grad are no longer needed, and will
    # harmlessly no-op if called by an existing training framework
    # opt.step()
    # opt.zero_grad()

# optionally remove gradient release hooks when done training
remove_gradient_release(model)
```

## Differences from PyTorch

optimi optimizers do not support compilation, differentiation, complex numbers, or have capturable versions.

optimi Adam optimizers do not support AMSGrad and SGD does not support Nesterov momentum. Optimizers which debias updates (Adam optimizers and Adan) calculate the debias term per parameter group, not per parameter.

## Optimizers

optimi implements the following optimizers: [Adam](optimizers/adam.md), [AdamW](optimizers/adamw.md), [Adan](optimizers/adan.md), [Lion](optimizers/lion.md), [RAdam](optimizers/radam.md), [Ranger](optimizers/ranger.md), [SGD](optimizers/sgd.md), & [StableAdamW](optimizers/stableadamw.md)