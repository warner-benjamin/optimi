---
title: "AdamW: Adam with Decoupled Weight Decay"
---

# AdamW: Adam with Decoupled Weight Decay

AdamW improves upon [Adam](adam.md) by decoupling weight decay from the gradients and instead applying weight decay directly to the model parameters. This modification allows AdamW to achieve better convergence and generalization than Adam.

AdamW was introduced by Ilya Loshchilov and Frank Hutter in [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101).

## Hyperparameters

optimi sets the default $\beta$s to `(0.9, 0.99)` and default $\epsilon$ to `1e-6`. These values reflect current best-practices and usually outperform the PyTorch defaults.

If training on large batch sizes or observing training loss spikes, consider reducing $\beta_2$ between $[0.95, 0.99)$.

optimi’s implementation of AdamW also supports [fully decoupled weight decay](../fully_decoupled_weight_decay.md) `decouple_lr=True`. The default weight decay of 0.01 will likely [need to be reduced](../fully_decoupled_weight_decay.md#hyperparameters) when using fully decoupled weight decay as the learning rate will not modify the effective weight decay.

::: optimi.adamw.AdamW

## Algorithm

Adam with decoupled weight decay (AdamW).

$$
\begin{aligned}
    &\rule{100mm}{0.4pt}\\
    &\hspace{2mm} \textcolor{#009ddb}{\textbf{Adam}} \: \textcolor{#9a3fe4}{\text{with decoupled weigh decay (AdamW)}} \\
    &\hspace{5mm} \text{inputs} : \bm{\theta}_0 \: \text{(params)}; \: f(\bm{\theta}) \text{(objective)}; \: \gamma_t \:\text{(learning rate at } t \text{)}; \\
    &\hspace{17.25mm} \beta_1, \beta_2 \: \text{(betas)}; \: \lambda \: \text{(weight decay)}; \: \epsilon \: \text{(epsilon)}\\
    &\hspace{5mm} \text{initialize} : \bm{m}_{0} \leftarrow \bm{0}; \: \bm{v}_{0} \leftarrow \bm{0}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
    &\hspace{5mm} \textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}\text{:}\\
        &\hspace{10mm} \bm{g}_t \leftarrow \nabla_{\theta} f_t(\bm{\theta}_{t-1}) \textcolor{#009ddb}{- \lambda\bm{\theta}_{t-1}}\\[0.5em]
        &\hspace{10mm} \bm{m}_t \leftarrow \beta_1 \bm{m}_{t-1} + (1 - \beta_1) \bm{g}_t\\
        &\hspace{10mm} \bm{v}_t \leftarrow \beta_2 \bm{v}_{t-1} + (1 - \beta_2) \bm{g}^2_t\\[0.5em]
        &\hspace{10mm} \hat{\bm{m}}_t \leftarrow \bm{m}_t/(1 - \beta_1^t)\\
        &\hspace{10mm} \hat{\bm{v}}_t \leftarrow \bm{v}_t/(1 - \beta_2^t)\\[0.5em]
        &\hspace{10mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \gamma_t \bigl( \hat{\bm{m}}_t / (\sqrt{\hat{\bm{v}}_t} + \epsilon) \textcolor{#9a3fe4}{+ \lambda\bm{\theta}_{t-1}} \bigr)\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
\end{aligned}
$$

optimi’s AdamW also supports [fully decoupled weight decay](../fully_decoupled_weight_decay.md#algorithm), which is not shown.