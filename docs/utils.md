---
title: Utilities
---

# Utilities

::: optimi.utils.param_groups_weight_decay

`param_groups_weight_decay` is adapted from [timm's optimizer factory methods](https://huggingface.co/docs/timm/reference/optimizers#timm.optim.create_optimizer).

### Example

`param_groups_weight_decay` takes a model and returns two optimizer parameter group dictionaries. One with bias and normalization terms without weight decay and another dictionary with the rest of the model parameters with weight decay. The `weigh_decay` passed to `param_groups_weight_decay` will override the optimizer's default weight decay.

```python
params = param_groups_weight_decay(model, weigh_decay=1e-5)
optimizer = StableAdamW(params, decouple_lr=True)

```

::: optimi.gradientrelease.prepare_for_gradient_release

For details on using `prepare_for_gradient_release`, please see the [gradient release docs](gradient_release.md).

::: optimi.gradientrelease.remove_gradient_release

For details on using `remove_gradient_release`, please see the [gradient release docs](gradient_release.md).