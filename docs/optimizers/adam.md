---
title: "Adam: Adaptive Moment Estimation"
---

# Adam: Adaptive Moment Estimation

Adam (Adaptive Moment Estimation) computes per-parameter adaptive learning rates from the first and second gradient moments. Adam combines the advantages of two other optimizers: AdaGrad, which adapts the learning rate to the parameters, and RMSProp, which uses a moving average of squared gradients to set per-parameter learning rates. Adam also introduces bias-corrected estimates of the first and second gradient averages.

Adam was introduced by Diederik Kingma and Jimmy Ba in [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980).

## Hyperparameters

optimi sets the default $\beta$s to `(0.9, 0.99)` and default $\epsilon$ to `1e-6`. These values reflect current best-practices and usually outperform the PyTorch defaults.

If training on large batch sizes or observing training loss spikes, consider reducing $\beta_2$ between $[0.95, 0.99)$.

optimi’s implementation of Adam combines Adam with both [AdamW](adamw.md) `decouple_wd=True` and Adam with [fully decoupled weight decay](../fully_decoupled_weight_decay.md) `decouple_lr=True`. Weight decay will likely [need to be reduced](../fully_decoupled_weight_decay.md#hyperparameters) when using fully decoupled weight decay as the learning rate will not modify the effective weight decay.

::: optimi.adam.Adam

## Algorithm

Adam with L2 regularization.

$$
\begin{aligned}
    &\rule{100mm}{0.4pt}\\
    &\hspace{2mm} \textbf{Adam}\\
    &\hspace{5mm} \text{inputs} : \bm{\theta}_0 \: \text{(params)}; \: f(\bm{\theta}) \text{(objective)}; \: \gamma_t \:\text{(learning rate at } t \text{)}; \\
    &\hspace{17.25mm} \beta_1, \beta_2 \: \text{(betas)}; \: \lambda \: \text{(weight decay)}; \: \epsilon \: \text{(epsilon)}\\
    &\hspace{5mm} \text{initialize} : \bm{m}_{0} \leftarrow \bm{0}; \: \bm{v}_{0} \leftarrow \bm{0}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
    &\hspace{5mm} \textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}\text{:}\\
        &\hspace{10mm} \bm{g}_t \leftarrow \nabla_{\theta} f_t(\bm{\theta}_{t-1}) - \lambda\bm{\theta}_{t-1}\\[0.5em]
        &\hspace{10mm} \bm{m}_t \leftarrow \beta_1 \bm{m}_{t-1} + (1 - \beta_1) \bm{g}_t\\
        &\hspace{10mm} \bm{v}_t \leftarrow \beta_2 \bm{v}_{t-1} + (1 - \beta_2) \bm{g}^2_t\\[0.5em]
        &\hspace{10mm} \hat{\bm{m}}_t \leftarrow \bm{m}_t/(1 - \beta_1^t)\\
        &\hspace{10mm} \hat{\bm{v}}_t \leftarrow \bm{v}_t/(1 - \beta_2^t)\\[0.5em]
        &\hspace{10mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \gamma_t \bigl( \hat{\bm{m}}_t / (\sqrt{\hat{\bm{v}}_t} + \epsilon) \bigr)\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
\end{aligned}
$$

optimi’s Adam also supports [AdamW’s](adamw.md#algorithm) decoupled weight decay and [fully decoupled weight decay](../fully_decoupled_weight_decay.md#algorithm), which are not shown.