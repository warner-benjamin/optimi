# optimī

### Fast, Modern, and Low Precision PyTorch Optimizers

optimi enables accurate low precision training via Kahan summation, supports fully decoupled weight decay, and features fast implementations of modern optimizers.

## Low Precision Training with Kahan Summation

optimi optimizers can match the performance of mixed precision when [training in BFloat16 by using Kahan summation](https://optimi.benjaminwarner.dev/kahan_summation).

Training in BFloat16 with Kahan summation can reduce non-activation training memory usage by [37.5 to 45.5 percent](https://optimi.benjaminwarner.dev/kahan_summation/#memory-savings) when using an Adam optimizer. BFloat16 training increases single GPU [training speed by ~10 percent](https://optimi.benjaminwarner.dev/kahan_summation/#training-speedup) at the same batch size.

## Fully Decoupled Weight Decay

In addition to supporting PyTorch-style decoupled weight decay, optimi optimizers also support [fully decoupled weight decay](https://optimi.benjaminwarner.dev/fully_decoupled_weight_decay).

Fully decoupled weight decay decouples weight decay from the learning rate, more accurately following [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101). This can help simplify hyperparameter tuning as the optimal weight decay is no longer tied to the learning rate.

## Foreach Implementations

All optimi optimizers have fast [foreach implementations](https://optimi.benjaminwarner.dev/foreach), which can significantly outperform the for-loop versions. optimi reuses the gradient buffer for temporary variables to reduce foreach memory usage.

## Documentation

<https://optimi.benjaminwarner.dev>

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

## Differences from PyTorch

optimi optimizers do not support compilation, differentiation, complex numbers, or have capturable versions.

optimi Adam optimizers do not support AMSGrad and SGD does not support Nesterov momentum. Optimizers which debias updates (Adam optimizers and Adan) calculate the debias term per parameter group, not per parameter.

## Optimizers

optimi implements the following optimizers: [Adam](https://optimi.benjaminwarner.dev/optimizers/adam), [AdamW](https://optimi.benjaminwarner.dev/optimizers/adamw), [Adan](https://optimi.benjaminwarner.dev/optimizers/adan), [Lion](https://optimi.benjaminwarner.dev/optimizers/lion), [RAdam](https://optimi.benjaminwarner.dev/optimizers/radam), [Ranger](https://optimi.benjaminwarner.dev/optimizers/ranger), [SGD](https://optimi.benjaminwarner.dev/optimizers/sgd), & [StableAdamW](https://optimi.benjaminwarner.dev/optimizers/stableadamw)
