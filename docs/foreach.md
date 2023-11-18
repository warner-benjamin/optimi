---
title: ForEach Optimizers
---

# ForEach Optimizer Implementations

Like PyTorch, optimi supports foreach implementations of all optimizers. Foreach optimizers can be significantly faster than the for-loop versions.

Foreach implementations can increase optimizer peak memory usage. optimi attempts to reduce this extra overhead by reusing the gradient buffer for temporary variables. If the gradients are required between the optimization step and [gradient reset step](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad), set `foreach=False` to use the for-loop implementation.

If unspecified `foreach=None`, optimi will use the foreach implementation if training on a Cuda device.