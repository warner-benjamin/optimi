---
title: "Optimizer Accumulation"
description: "Gradient Release with Approximate Gradient Accumulation"
---

# Optimizer Accumulation

**Gradient Release with Approximate Gradient Accumulation**

[Gradient accumulation](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation) reduces training memory by splitting a batch into micro-batches and accumulating micro-batch gradients into the larger batch. [Gradient release](gradient_release.md) reduces training memory by limiting gradients to one layer at any given time. Optimizer accumulation unifies these two disparate approaches by accumulating gradients directly into optimizer states while performing gradient release.

During the backward pass, each model layer calculates its gradients, performs a partial optimizer step, and clears the gradients before proceeding to the backward pass for the next layer. The partial optimizer step accumulates gradients by updating the optimizer state but not modifying the model weights. After multiple gradients have been accumulated into optimizer states, a normal optimizer step is ran updating the model parameters with the accumulated states.

Optimizer accumulation can reduce non-activation memory usage by ~40 percent compared to an Adam optimizer with gradient accumulation. Optimizer accumulation can also be combined with other techniques such as [Kahan summation](kahan_summation.md) or [activation checkpointing](https://pytorch.org/docs/stable/checkpoint.html) for further memory savings.

??? note "Note: Optimizer Accumulation Requires PyTorch 2.1+"

    Optimizer accumulation requires PyTorch 2.1 or newer.

Optimizer accumulation was proposed by Zhang et al in [*AdamAccumulation to Reduce Memory Footprints of both Activations and Gradients for Large-scale DNN Training*](https://arxiv.org/abs/2305.19982). optimi’s implementation enables AdamAccumulation for all optimi optimizers[^1].

Zhang et al report that models trained with an AdamAccumulation over eight micro-batches match models trained via Adam with gradient accumulation over eight micro-batches.

## Limitations and Workarounds

Since optimizer accumulation immediately frees the gradient during the backward pass, features which rely on persistent gradients like AMP's `GradScaler`, gradient clipping, or gradient accumulation won’t work. L2 weight decay also shouldn’t be used with optimizer accumulation.

!!! warning "Important: Optimizer Accumulation is Incompatible with FP16 Mixed Precision"

    Optimizer accumulation is incompatible with Float16 Automatic Mixed Precision since PyTorch's `GradScaler` requires access to the entire model's gradients for the optimizer step.

    Use BFloat16 Automatic Mixed Precision instead.

The recommended workaround for gradient clipping is to use [StableAdamW](optimizers/stableadamw.md) instead of Adam or AdamW, as StableAdamW removes the need for gradient clipping by porting Adafactor’s update clipping into AdamW.

!!! warning "Important: Don't use L2 Weight Decay with Optimizer Accumulation"

    optimi applies weight decay on the full optimization step. Since L2 weight decay operates on the gradients, it would only be applied on the last gradient instead of all gradients.

    Use decoupled or [fully decoupled weight decay](fully_decoupled_weight_decay.md) instead.

Because the gradients are accumulated into the optimizer states, applying beta and momentum terms, optimizer accumulation approximates gradient accumulation.

## Example

Using optimi’s optimizer accumulation requires three steps: initializing the optimizer with `gradient_release=True`, calling `prepare_for_gradient_release` on both the model and optimizer, and setting `optimizer.optimizer_accumulation` to True or False to accumulation gradients or perform a full optimizer step, respectively.

Like gradient accumulation, set `optimizer_accumulation=True` before the backward step while accumulating gradients and `optimizer_accumulation=False` when model parameters are to be updated by the full optimizer step.

```python
import torch
from torch import nn
from optimi import AdamW

# create or cast model in low precision (bfloat16)
model = nn.Linear(20, 1, dtype=torch.bfloat16)

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

[^1]: While optimizer accumulation is noisy compared to gradient accumulation, SGD's optimizer accumulation results are significantly nosier then all other optimizers.