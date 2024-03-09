---
title: "Gradient Release: Fused Backward and Optimizer Step"
---

# Gradient Release: Fused Backward and Optimizer Step

Gradient release reduces training memory by limiting gradients to one layer at any given time. Unlike [PyTorch’s implementation](https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html), optimi’s gradient release is fully compatible with both existing learning rate and optimizer schedulers and existing training frameworks.

During the backward pass, each model layer calculates its gradients, performs the optimizer step, and clears the gradients before proceeding to the backward pass for the next layer. This fused backward and optimizer step can reduce non-activation memory usage by ~25 percent for an Adam optimizer.

Gradient release can also be combined with other techniques such as [Kahan summation](kahan_summation.md) or [activation checkpointing](https://pytorch.org/docs/stable/checkpoint.html) for further memory savings.

??? note "Note: Gradient Release Requires PyTorch 2.1+"

    Gradient release requires PyTorch 2.1 or newer.

Gradient release was proposed by Pudipeddi et al in [*Training Large Neural Networks with Constant Memory using a New Execution Algorithm*](https://arxiv.org/abs/2002.05645) and was enabled by PyTorch’s [`register_post_accumulate_grad_hook`](https://pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html).

## Limitations and Workarounds

Since gradient release immediately frees the gradient during the backward pass, features which rely on persistent gradients like AMP's `GradScaler`, gradient clipping, or gradient accumulation won’t work.

!!! warning "Important: Gradient Release is Incompatible with FP16 Mixed Precision"

    Gradient release is incompatible with Float16 Automatic Mixed Precision since PyTorch's `GradScaler` requires access to the entire model's gradients for the optimizer step.

    Use BFloat16 Automatic Mixed Precision instead.

The recommended workaround for gradient clipping is to use [StableAdamW](optimizers/stableadamw.md) instead of Adam or AdamW, as StableAdamW removes the need for gradient clipping by porting Adafactor’s update clipping into AdamW.

??? tip "Tip: Use Optimizer Accumulation to Approximate Gradient Accumulation"

    optimi's [optimizer accumulation](optimizer_accumulation.md) approximates [gradient accumlation](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation) by defering parameter updates while accumulating gradients directly into the optimizer states.

One potential workaround for gradient accumulation is to increase the optimizer’s momentum or $\beta_1$ to approximate accumulating gradients across multiple batches.

## Example

Using optimi’s gradient release requires two steps: initializing the optimizer with `gradient_release=True` and calling `prepare_for_gradient_release` on both the model and optimizer.

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