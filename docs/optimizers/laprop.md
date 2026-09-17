---
title: "LaProp: Separating Momentum and Adaptivity in Adam"
---

# LaProp: Separating Momentum and Adaptivity in Adam

LaProp modifies [Adam](adam.md) by normalizing the gradient by the square root of the debiased second moment before computing the first moment. This prevents changes in the second moment from rescaling the previously accumulated first moment, improving robustness to sudden changes in gradients. LaProp also weights contributions to the first moment by their learning rates, so gradients accumulated at higher learning rates retain more influence relative to newer gradients as the learning rate decays.

LaProp was introduced by Liu et al in [*LaProp: a Better Way to Combine Momentum with Adaptive Gradient*](https://arxiv.org/abs/2002.04839).

## Hyperparameters

optimi sets the default $\beta$s to `(0.9, 0.99)` and default $\epsilon$ to `1e-6`. These values reflect current best-practices. Liu et al default to $\beta$s `(0.9, 0.999)` and $\epsilon$ `1e-15`.

If training on large batch sizes or observing training loss spikes, consider reducing $\beta_2$ between $[0.95, 0.99)$.

By default, LaProp applies decoupled weight decay after the parameter update. Set `adam_wd=True` to apply weight decay before the update (AdamW-style).

optimi's implementation of LaProp also supports [fully decoupled weight decay](../fully_decoupled_weight_decay.md) `decouple_lr=True`. Remember that weight decay will likely [need to be reduced](../fully_decoupled_weight_decay.md#hyperparameters) when using fully decoupled weight decay as the learning rate will not modify the effective weight decay.

::: optimi.laprop.LaProp

## Algorithm

LaProp with decoupled weight decay.

$$
\begin{aligned}
    &\rule{100mm}{0.4pt}\\
    &\hspace{2mm} \textbf{LaProp} \\
    &\hspace{5mm} \text{inputs} : \bm{\theta}_0 \: \text{(params)}; \: f(\bm{\theta}) \text{(objective)}; \: \gamma_t \:\text{(learning rate at } t \text{)}; \\
    &\hspace{17.25mm} \beta_1, \beta_2 \: \text{(betas)}; \: \lambda \: \text{(weight decay)}; \: \epsilon \: \text{(epsilon)}\\
    &\hspace{5mm} \text{initialize} : \bm{m}_{0} \leftarrow \bm{0}; \: \bm{v}_{0} \leftarrow \bm{0}; \: \textcolor{#9a3fe4}{\overline{\gamma}_0 \leftarrow 0}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
    &\hspace{5mm} \textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}\text{:}\\
        &\hspace{10mm} \bm{g}_t \leftarrow \nabla_{\theta} f_t(\bm{\theta}_{t-1})\\[0.5em]
        &\hspace{10mm} \textcolor{#9a3fe4}{\hat{\beta}_{2,t} \leftarrow \beta_2(1 - \beta_2^{t-1})/(1 - \beta_2^t)}\\
        &\hspace{10mm} \textcolor{#9a3fe4}{\bm{v}_t \leftarrow \hat{\beta}_{2,t} \bm{v}_{t-1} + (1 - \hat{\beta}_{2,t}) \bm{g}^2_t}\\[0.5em]
        &\hspace{10mm} \textcolor{#9a3fe4}{\overline{\gamma}_t \leftarrow \beta_1 \overline{\gamma}_{t-1} + (1 - \beta_1) \gamma_t}\\
        &\hspace{10mm} \textcolor{#9a3fe4}{\hat{\beta}_{1,t} \leftarrow \beta_1 \overline{\gamma}_{t-1} / \overline{\gamma}_t}\\[0.5em]
        &\hspace{10mm} \textcolor{#9a3fe4}{\bm{m}_t \leftarrow \hat{\beta}_{1,t} \bm{m}_{t-1} + (1 - \hat{\beta}_{1,t}) \bm{g}_t / (\sqrt{\bm{v}_t} + \epsilon)}\\[0.5em]
        &\hspace{10mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \gamma_t \bm{m}_t\\
        &\hspace{10mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t} - \gamma_t \lambda\bm{\theta}_{t}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
\end{aligned}
$$

optimi's LaProp also supports [fully decoupled weight decay](../fully_decoupled_weight_decay.md#algorithm) and Adam-style weight decay, which are not shown.
