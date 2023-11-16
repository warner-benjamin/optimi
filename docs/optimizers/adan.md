---
title: "Adan: ADAptive Nesterov Momentum"
---

# Adan: ADAptive Nesterov Momentum

Adan uses a efficient Nesterov momentum estimation method to avoid the extra computation and memory overhead of calculating the extrapolation point gradient. In contrast to other Nesterov momentum estimating optimizers, Adan estimates both the first- and second-order gradient movements. This estimation requires two additional buffers over [AdamW](adamw.md), increasing memory usage.

Adan was introduced by Xie et al in *[Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models](https://arxiv.org/abs/2208.06677)*.

## Hyperparameters

Hyperparameter notes from Xie et al:

1. $\beta_2$ is the least sensitive Adan hyperparameter, default of 0.92 works for majority of tasks
2. Xie et al primarily tune $\beta_3$ (between 0.9-0.999) before $\beta_1$ (between 0.9-0.98) for different tasks
3. Adan pairs well with large learning rates. Paper and GitHub report up to 3x larger than Lamb and up to 5-10x larger than [AdamW](adamw.md)
4. Xie et al use the default weight decay of 0.02 for all tasks except fine-tuning BERT (0.01) and reinforcement learning (0)

optimi’s implementation of Adan also supports [fully decoupled weight decay](../fully_decoupled_weight_decay.md) `decouple_lr=True`. The default weight decay of 0.02 will likely [need to be reduced](../fully_decoupled_weight_decay.md#hyperparameters) when using fully decoupled weight decay as the learning rate will not modify the effective weight decay.

::: optimi.adan.Adan

## Algorithm

Adan: Adaptive Nesterov Momentum.

$$
\begin{align*}
    &\rule{100mm}{0.4pt}\\
    &\hspace{2mm} \textbf{Adan}  \\
    &\hspace{5mm} \text{inputs} : \bm{\theta}_0 \: \text{(params)}; \: f(\bm{\theta}) \text{(objective)}; \: \gamma_t \:\text{(learning rate at } t \text{)}; \\
    &\hspace{17.25mm} \beta_1, \beta_2, \beta_3 \: \text{(betas)}; \: \lambda \: \text{(weight decay)}; \: \epsilon \text{ (epsilon)}\\
    &\hspace{5mm} \text{initialize} : \bm{m}_{0} \leftarrow \bm{0}; \: \bm{v}_{0} \leftarrow \bm{0}; \: \bm{n}_{0} \leftarrow \bm{0}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
    &\hspace{5mm} \textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}\text{:}\\
        &\hspace{10mm} \bm{g}_t \leftarrow \nabla_{\theta} f_t(\bm{\theta}_{t-1})\\[0.5em]
        &\hspace{10mm} \bm{m}_t \leftarrow \beta_1 \bm{m}_{t-1} + (1 - \beta_1) \bm{g}_t\\
        &\hspace{10mm} \bm{v}_t \leftarrow \beta_2 \bm{v}_{t-1} + (1 - \beta_2) (\bm{g}_t - \bm{g}_{t-1})\\
        &\hspace{10mm} \bm{n}_t \leftarrow \beta_3 \bm{n}_{t-1} + (1 - \beta_3)\bigl(\bm{g}_t + \beta_2(\bm{g}_t - \bm{g}_{t-1})\bigr)^2\\[0.5em]
        &\hspace{10mm} \hat{\bm{m}}_t \leftarrow \bm{m}_t/(1 - \beta_1^t)\\
        &\hspace{10mm} \hat{\bm{v}}_t \leftarrow \bm{v}_t/(1 - \beta_2^t)\\
        &\hspace{10mm} \hat{\bm{n}}_t \leftarrow \bm{n}_t/(1 - \beta_3^t)\\[0.5em]
        &\hspace{10mm} \bm{\eta}_t \leftarrow \gamma_t/(\sqrt{\hat{\bm{n}}_t} + \epsilon)\\
        &\hspace{10mm} \bm{\theta}_t \leftarrow (1+\gamma_t\lambda )^{-1}\bigl(\bm{\theta}_{t-1} - \bm{\eta}_t (\hat{\bm{m}}_t + \beta_2\hat{\bm{v}}_t)\bigr)\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
\end{align*}
$$

During the first step, $\bm{g}_t - \bm{g}_{t-1}$ is set to $\bm{0}$.

optimi’s Adan also supports Adam-style weight decay and [fully decoupled weight decay](../fully_decoupled_weight_decay.md#algorithm), both which are not shown.