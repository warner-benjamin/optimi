---
title: "RAdam: Rectified Adam"
---

# RAdam: Rectified Adam

RAdam (Rectified Adam) is a variant of Adam which improves Adam’s convergence by fixing the adaptive learning rate's large variance during early stages of training. RAdam estimates the variance of the squared gradient moving average and scales the update by this term to rectify the variance. RAdam is comparable to using a learning rate warmup schedule.

RAdam was introduced by Liu et al in [*On the Variance of the Adaptive Learning Rate and Beyond*](https://arxiv.org/abs/1908.03265).

## Hyperparameters

optimi sets the default $\beta$s to `(0.9, 0.99)` and default $\epsilon$ to `1e-6`. These values reflect current best-practices and usually outperform the PyTorch defaults.

If training on large batch sizes or observing training loss spikes, consider reducing $\beta_2$ between $[0.95, 0.99)$.

optimi’s implementation of RAdam supports both [decoupled weight decay](adamw.md) `decouple_wd=True` and [fully decoupled weight decay](../fully_decoupled_weight_decay.md) `decouple_lr=True`. Weight decay will likely [need to be reduced](../fully_decoupled_weight_decay.md#hyperparameters) when using fully decoupled weight decay as the learning rate will not modify the effective weight decay.

::: optimi.radam.RAdam

## Algorithm

RAdam: Rectified Adam.

$$
\begin{aligned}
    &\rule{100mm}{0.4pt}\\
    &\hspace{2mm} \textcolor{#9a3fe4}{\textbf{Rectified}} \: \textbf{Adam}\\
    &\hspace{5mm} \text{inputs} : \bm{\theta}_0 \: \text{(params)}; \: f(\bm{\theta}) \text{(objective)}; \: \gamma_t \:\text{(learning rate at } t \text{)}; \\
    &\hspace{17.25mm} \beta_1, \beta_2 \: \text{(betas)}; \: \lambda \: \text{(weight decay)}; \: \epsilon \: \text{(epsilon)};\\
    &\hspace{5mm} \text{initialize} : \bm{m}_{0} \leftarrow \bm{0}; \: \bm{v}_{0} \leftarrow \bm{0}; \: \textcolor{#9a3fe4}{\rho_{\infty} \leftarrow 2 / (1 - \beta_2) - 1}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
    &\hspace{5mm} \textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}\text{:}\\
        &\hspace{10mm} \bm{g}_t \leftarrow \nabla_{\theta} f_t(\bm{\theta}_{t-1}) - \lambda\bm{\theta}_{t-1}\\[0.5em]
        &\hspace{10mm} \bm{m}_t \leftarrow \beta_1 \bm{m}_{t-1} + (1 - \beta_1) \bm{g}_t\\
        &\hspace{10mm} \bm{v}_t \leftarrow \beta_2 \bm{v}_{t-1} + (1 - \beta_2) \bm{g}^2_t\\[0.5em]
        &\hspace{10mm} \hat{\bm{m}}_t \leftarrow \bm{m}_t/(1 - \beta_1^t)\\
        &\hspace{10mm} \hat{\bm{v}}_t \leftarrow \bm{v}_t/(1 - \beta_2^t)\\[0.5em]
        &\hspace{10mm} \textcolor{#9a3fe4}{\rho_t \leftarrow \rho_{\infty} - 2 t \beta^t_2 /(1 - \beta_2^t)}\\[0.5em]
        &\hspace{10mm} \textcolor{#9a3fe4}{\textbf{if} \: \rho_t > 5\text{:}}\\
        &\hspace{15mm} \textcolor{#9a3fe4}{r_t \leftarrow \sqrt{\tfrac{(\rho_t - 4)(\rho_t - 2)\rho_{\infty}}{(\rho_{\infty} - 4)(\rho_{\infty} -2 ) \rho_t}}}\\
        &\hspace{15mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \gamma_t \textcolor{#9a3fe4}{r_t} \bigl( \hat{\bm{m}}_t / (\sqrt{\hat{\bm{v}}_t} + \epsilon) \bigr)\\
        &\hspace{10mm} \textcolor{#9a3fe4}{\textbf{else}\text{:}}\\
        &\hspace{15mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \gamma_t \textcolor{#9a3fe4}{\hat{\bm{m}}_t}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
\end{aligned}
$$

optimi’s RAdam also supports [decoupled weight decay](adamw.md#algorithm) and [fully decoupled weight decay](../fully_decoupled_weight_decay.md#algorithm), which are not shown.