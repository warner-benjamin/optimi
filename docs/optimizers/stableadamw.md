---
title: "StableAdamW: AdamW with Update Clipping"
---

# StableAdamW: AdamW with Update Clipping

StableAdamW is a [AdamW](https://openreview.net/forum?id=Bkg6RiCqY7)-[Adafactor](https://proceedings.mlr.press/v80/shazeer18a.html) hybrid, porting Adafactor’s update clipping into [AdamW](adamw.md) as a per parameter learning rate modification. StableAdamW’s update clipping outperforms gradient clipping on downstream tasks while avoiding model training instability.

StableAdamW was introduced by Wortsman et al in *[Stable and low-precision training for large-scale vision-language models](https://arxiv.org/abs/2304.13013)*.

## Hyperparameters

StableAdamW is a drop-in replacement for AdamW and uses the same hyperparameters, with one exception: StableAdamW removes the need for gradient clipping.

If training on large batch sizes or still observing training loss spikes, consider reducing $\beta_2$ between $[0.95, 0.99)$.

optimi’s implementation of StableAdamW also supports [fully decoupled weight decay](../fully_decoupled_weight_decay.md) `decouple_lr=True`. The default weight decay of 0.01 will likely [need to be reduced](../fully_decoupled_weight_decay.md#hyperparameters) when using fully decoupled weight decay as the learning rate will not modify the effective weight decay.

::: optimi.stableadamw.StableAdamW

## Algorithm

StableAdam with decoupled weight decay (StableAdamW).

$$
\begin{aligned}
    &\rule{100mm}{0.4pt}\\
    &\hspace{2mm} \textbf{\textcolor{#9a3fe4}{Stable}AdamW} \\
    &\hspace{5mm} \text{inputs} : \bm{\theta}_0 \: \text{(params)}; \: f(\bm{\theta}) \text{(objective)}; \: \gamma_t \:\text{(learning rate at } t \text{)}; \\
    &\hspace{17.25mm} \beta_1, \beta_2 \: \text{(betas)}; \: \lambda \: \text{(weight decay)}; \: \epsilon \: \text{(epsilon)}\\
    &\hspace{5mm} \text{initialize} : \bm{m}_{0} \leftarrow \bm{0}; \: \bm{v}_{0} \leftarrow \bm{0}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
    &\hspace{5mm} \textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}\text{:}\\
        &\hspace{10mm} \bm{g}_t \leftarrow \nabla_{\theta} f_t(\bm{\theta}_{t-1})\\[0.5em]
        &\hspace{10mm} \bm{m}_t \leftarrow \beta_1 \bm{m}_{t-1} + (1 - \beta_1) \bm{g}_t\\
        &\hspace{10mm} \bm{v}_t \leftarrow \beta_2 \bm{v}_{t-1} + (1 - \beta_2) \bm{g}^2_t\\[0.5em]
        &\hspace{10mm} \hat{\bm{m}}_t \leftarrow \bm{m}_t/(1 - \beta_1^t)\\
        &\hspace{10mm} \hat{\bm{v}}_t \leftarrow \bm{v}_t/(1 - \beta_2^t)\\[0.5em]
        &\hspace{10mm} \textcolor{#9a3fe4}{\textbf{RMS}_t \leftarrow  \sqrt{\mathbb{E[\bm{g}^2_t/\text{max}(\bm{v}_t, \epsilon^2)]}}}\\
        &\hspace{10mm} \textcolor{#9a3fe4}{\bm{\eta}_t \leftarrow  \gamma_t/\text{max}(1,\textbf{RMS}_t)}\\[0.5em]
        &\hspace{10mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \textcolor{#9a3fe4}{\bm{\eta}_t} \bigl( \hat{\bm{m}}_t / (\sqrt{\hat{\bm{v}}_t} + \epsilon) + \lambda\bm{\theta}_{t-1} \bigr)\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
\end{aligned}
$$

Following *[Stable and low-precision training for large-scale vision-language models](https://arxiv.org/abs/2304.13013)*, the $\text{RMS}_t$ steps occur independantly for each tensor. Likewise, the $\text{max}(\bm{v}_t, \epsilon^2)$ term, instead of $\sqrt{\mathbb{E[\bm{g}^2_t/\bm{v}_t]}}$, is added to prevent division by zero issues.

optimi’s StableAdamW also supports [fully decoupled weight decay](../fully_decoupled_weight_decay.md#algorithm), which is not shown.