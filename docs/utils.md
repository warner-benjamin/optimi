---
title: Utilities
---

# Utilities

::: optimi.utils.param_groups_weight_decay

`param_groups_weight_decay` is adapted from [timm's optimizer factory methods](https://huggingface.co/docs/timm/reference/optimizers#timm.optim.create_optimizer).

### Examples

`param_groups_weight_decay` takes a model and returns two optimizer parameter group dictionaries. One with bias and normalization terms without weight decay and another dictionary with the rest of the model parameters with weight decay. The `weight_decay` passed to `param_groups_weight_decay` will override the optimizer's default weight decay.

```python
params = param_groups_weight_decay(model, weight_decay=1e-5)
optimizer = StableAdamW(params, decouple_lr=True)

```

`additional_layers` parameter allows you to specify additional layer names or name substrings that should be excluded from weight decay. This is useful for excluding specific layers like token embeddings which also benefit from not having weight decay applied.

The parameter accepts an iterable of strings, where each string is matched as a substring against the full parameter name (as returned by `model.named_parameters()`).

```python
class MiniLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(1000, 20)
        self.pos_embeddings = nn.Embedding(100, 20)
        self.norm = nn.LayerNorm(20)
        self.layer1 = nn.Linear(20, 30)
        self.layer2 = nn.Linear(30, 1000)

model = MiniLM()

# Exclude token embeddings from weight decay in addition to bias and normalization layers
params = param_groups_weight_decay(
    model,
    weight_decay=1e-5,
    additional_layers=["tok_embeddings"]
)
```

::: optimi.utils.to_low_precision

### Examples

```python
from torch import nn
from optimi.utils import to_low_precision

model = nn.Sequential(
    nn.Embedding(1000, 128),
    nn.Linear(128, 128),
)

to_low_precision(model, dtype=torch.bfloat16)
```

Override which modules remain float32 (e.g., only keep `LayerNorm` in float32):

```python
to_low_precision(model, dtype=torch.bfloat16, fp32_modules=(nn.LayerNorm,))
```

Disable RoPE/rotary buffer preservation so they are also cast to low precision:

```python
to_low_precision(model, dtype=torch.bfloat16, fp32_buffers=None)
```

Cast and move to a device in one pass:

```python
to_low_precision(model, dtype=torch.bfloat16, device="cuda")
```

::: optimi.gradientrelease.prepare_for_gradient_release

For details on using `prepare_for_gradient_release`, please see the [gradient release docs](gradient_release.md).

::: optimi.gradientrelease.remove_gradient_release

For details on using `remove_gradient_release`, please see the [gradient release docs](gradient_release.md).