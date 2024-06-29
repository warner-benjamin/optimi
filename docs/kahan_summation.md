---
title: Low Precision Training with Kahan Summation
---

# Low Precision Training with Kahan Summation

While training models in low precision (Float16 or BFloat16) usually differs from training in full precision (Float32) or [mixed precision](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch), optimi optimizers nearly reach or match the performance of mixed precision when training in BFloat16 by using Kahan summation[^1].

Training in low precision [reduces non-activation memory usage up to ~46 percent](#memory-savings) and increases [training speed up to ~30 percent](#training-speedup) relative to mixed precision training.

Using Kahan summation for accurate BFloat16 training is as simple as replacing a PyTorch optimizer with its optimi equivalent and casting the model to BFloat16 instead of using mixed precision.

!!! tip "Tip: Keep a Few Layers in Float32"

    When training in BFloat16, keep rotary embedding layers in Float32 and [consider keeping normalization layers](https://docs.mosaicml.com/projects/composer/en/latest/method_cards/low_precision_layernorm.html) in Float32, as these layers can benefit from full precision. This results in a small memory increase and speed decrease but can help guarantee equivalent results with mixed precision training.

By default, optimi optimizers will automatically use Kahan summation for any layers training in low precision. Set `kahan_sum=False` to disable.

## Mixed Precision

While implementations details can differ, mixed precision works by running a forward pass in low precision, automatically switching to full precision per layer as needed, and then accumulating gradients during the backward pass in Float32. The optimizer step runs in full precision.

The hybrid precision setup of mixed precision enables the faster operations and lower memory usage of low precision while keeping the convergence of full precision.

## Kahan Summation

[Kahan summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)[^2] is a technique to reduce the numerical error of adding multiple low precision numbers by accumulating errors in a separate compensation buffer. The addition of the compensation buffer increases the effective summation precision by the precision of the compensation buffer.

Using Kahan summation to improve low precision model training was first introduced by Zamirai et al in [*Revisiting BFloat16 Training*](https://arxiv.org/abs/2010.06192). Zamirai et al discovered the primary source of numerical error from low precision training is during the optimizer’s model weight update step. They add Kahan summation to the SGD & AdamW weight update steps to reduce the update’s numerical inaccuracy, increasing low precision training to the equivalent of full precision training across tested models.

??? note "Note: Implementation Inspired by TorchDistX"

    optimi’s Kahan summation implementation was directly inspired by [TorchDistX’s](https://github.com/pytorch/torchdistx) `AnyPrecisionAdamW` optimizer.

For more details, see the [algorithm](#algorithm) and [explanation](# explanation) sections.

## Memory Savings

Training in BFloat16 with Kahan summation can reduce non-activation training memory usage by 37.5 to 46.2 percent when using an Adam optimizer, as Table 1 shows below.

Table: Adam Per Parameter Memory Usage, Excluding Activations

| Buffer | Mixed Precision | BFloat16 + Kahan Sum | BFloat16 |
|:----|:---:|:---:|:---:|
| BF16 Model Weights (if used) | 2 bytes | 2 bytes | 2 bytes |
| FP32 Model Weights | 4 bytes  | - | - |
| Gradients | 4 bytes | 2 bytes | 2 bytes |
| Gradient Accumulation | 4 bytes | 2 bytes | 2 bytes |
| Distributed Training | 4 bytes | 2 bytes | 2 bytes |
| Momentum  | 4 bytes | 2 bytes | 2 bytes |
| Variance  | 4 bytes | 2 bytes | 2 bytes |
| Kahan Compensation | - | 2 bytes | - |
| **Total**  | 16-26 bytes | 8-14 bytes | 6-12 bytes |

Calculating the total memory savings depends on [activations and batch size](https://blog.eleuther.ai/transformer-math/#activations-and-batch-size), mixed precision implementation details, and the optimizer used, to name a few variables.

optimi reduces potential extra memory overhead of Kahan summation by reusing the gradient buffer for temporary variables.

## Training Speedup

Training in BFloat16 instead of mixed precision results in a ~10% speedup on a single GPU, ~20% speedup with two GPUs, and up to ~30% speedup with multiple GPUs[^3].

## Example

Using Kahan summation with an optimi optimizer only requires a casting a model and optionally input into low precision (BFloat16 or Float16). Since Kahan summation is applied layer by layer, it works for models with standard and low precision weights.

```python
import torch
from torch import nn
from optimi import AdamW

# create or cast some model layers in low precision (bfloat16)
model = nn.Linear(20, 1, dtype=torch.bfloat16)

# initialize any optmi optimizer with low precsion parameters
# Kahan summation is enabled since some model layers are bfloat16
opt = AdamW(model.parameters(), lr=1e-3)

# forward and backward, casting input to bfloat16 if needed
loss = model(torch.randn(20, dtype=torch.bfloat16))
loss.backward()

# optimizer step automatically uses Kahan summation for low precision layers
opt.step()
opt.zero_grad()
```

To disable Kahan Summation pass `kahan_summation=False` on optimizer initialization.

## Algorithm

SGD with Kahan summation.

$$
\begin{aligned}
    &\rule{90mm}{0.4pt}\\
    &\hspace{2mm} \textcolor{#009ddb}{\textbf{SGD}} \: \textcolor{#9a3fe4}{\text{with Kahan summation}}\\
    &\hspace{5mm} \text{inputs} : \bm{\theta}_0 \: \text{(params)}; \: f(\bm{\theta}) \text{(objective)};\\
    &\hspace{17.25mm} \gamma_t \:\text{(learning rate at } t \text{)}; \: \lambda \: \text{(weight decay)}\\
    &\hspace{5mm} \text{initialize} : \textcolor{#9a3fe4}{\bm{k}_{0} \leftarrow \bm{0}}\\[-0.5em]
    &\rule{90mm}{0.4pt}\\
    &\hspace{5mm} \textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}\text{:}\\
        &\hspace{10mm} \bm{g}_t \leftarrow \nabla_{\theta} f_t(\bm{\theta}_{t-1}) - \lambda\bm{\theta}_{t-1}\\[0.5em]
        &\hspace{10mm} \textcolor{#009ddb}{\bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \gamma_t\bm{g}_t}\\[0.3em]
        &\hspace{10mm} \textcolor{#9a3fe4}{\bm{u}_t \leftarrow \bm{k}_{t-1} - \gamma_t\bm{g}_t}\\
        &\hspace{10mm} \textcolor{#9a3fe4}{\bm{\theta}_t \leftarrow \bm{\theta}_{t-1} + \bm{u}_t}\\
        &\hspace{10mm} \textcolor{#9a3fe4}{\bm{k}_t \leftarrow \bm{u}_t + (\bm{\theta}_{t-1} - \bm{\theta}_t)}\\[-0.5em]
    &\rule{90mm}{0.4pt}\\
\end{aligned}
$$

This shows the optimi implementation of Kahan summation optimizers, which is equivalent to the *Revisiting BFloat16 Training* formulation.

## Explanation

optimi optimizers with Kahan summation modify the base optimizers by introducing a compensation buffer $k$ to mitigate numerical errors from training in low precision.

Using SGD as an example, the SGD parameter update is straightforward: $\textcolor{#009ddb}{\bm{\theta}_t ← \bm{\theta}_{t-1} - \gamma_t\bm{g}_t}$. Where $\theta$ is the model parameters at steps $t-1$ and $t$, and $\gamma_t$ and $g_t$ are the learning rate and gradient at step $t$, respectively.

SGD with Kahan summation expands the single update model parameter step to three steps:

1. $\textcolor{#9a3fe4}{\bm{u}_t ← \bm{k}_{t-1} - \gamma_t\bm{g}_t}$: First, an intermediate update $u_t$ is computed from the prior compensation buffer $k_{t-1}$. This allows the current parameter update to account for any precision errors in last parameter update.
2. $\textcolor{#9a3fe4}{\bm{\theta}_t ← \bm{\theta}_{t-1} + \bm{u}_t}$: The parameter update uses the error compensated update $u_t$ instead of directly subtracting $\gamma_t g_t$.
3. $\textcolor{#9a3fe4}{\bm{k}_t ← \bm{u}_t + (\bm{\theta}_{t-1} - \bm{\theta}_t)}$: Finally, the compensation buffer is updated to stores the difference between the intended update ($u_t$) and the actual change in the parameters $(\theta_{t-1} - \theta_t)$, capturing any errors from operating in low precision numerical types.

These Kahan summation steps allow optimi optimizers to nearly reach or match the performance of mixed precision when training in low precision.

[^1]: Current testing on small models shows little to no degradation in model performance.

[^2]: Also known as Kahan–Babuška summation or compensated summation.

[^3]: BFloat16 training increases distributed training speed more then single GPU due to the halved bandwidth cost. Observed results may differ based on GPU connectivity.