---
title: Fully Decoupled Weight Decay
---

# Fully Decoupled Weight Decay

In addition to supporting PyTorch-style decoupled weight decay, optimi optimizers also support fully decoupled weight decay.

While PyTorch-style decoupled weight decay (hereafter referred to as “decoupled weight decay”) decouples weight decay from the gradients, it explicitly couples weight decay with the learning rate $\gamma_t\lambda\bm{\theta}_{t-1}$. This ties the optimal value of weight decay to the learning rate.

In contrast, optimi’s fully decoupled weight decay also decouples weight decay from the learning rate, more accurately following [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) by Loshchilov and Hutter.

Fully decoupled weight decay is scaled by the relative learning rate $(\gamma_t/\gamma_\text{max})\lambda\bm{\theta}_{t-1}$ so applied weight decay will still follow the learning rate schedule.

??? note "Note: Implementation Inspired by Composer"

    optimi’s fully decoupled weight decay implementation was inspired by Mosaic Composer’s [Decoupled Weight Decay](https://docs.mosaicml.com/projects/composer/en/stable/method_cards/decoupled_weight_decay.html implementation).

By default, optimi optimizers do not use fully decoupled weight decay for compatibility with existing PyTorch code.

Enable fully decoupled weight decay by setting `decouple_lr=True` when initialing an optimi optimizer. If the initial learning rate `lr` isn’t the maximum scheduled learning rate, pass it to `max_lr`.

## Hyperparameters

Since fully decoupled weight decay is not multiplied by the learning rate each step, the optimal value for fully decoupled weight decay is smaller than decoupled weight decay.

For example, to match [AdamW’s](optimizers/adamw.md) default decoupled weight decay of 0.01 with a maximum learning rate of $1\times10^{-3}$, set weight decay to $1\times10^{-5}$ when using fully decoupled weight decay.

By default, optimi optimizers assume `lr` is the maximum scheduled learning rate. This allows the applied weight decay $(\gamma_t/\gamma_\text{max})\lambda\bm{\theta}_{t-1}$ to match the learning rate schedule. Set `max_lr` if this is not the case.

## Algorithm

The algorithm below shows the difference between PyTorch’s AdamW and optimi’s Adam with fully decoupled weight decay.


$$
\begin{aligned}
    &\rule{105mm}{0.4pt}\\
    &\hspace{2mm} \textcolor{#009ddb}{\text{PyTorch’s AdamW}} \: \text{\&} \: \textcolor{#9a3fe4}{\text{Adam with fully decoupled weight decay}}\\
    &\hspace{5mm} \text{inputs} : \bm{\theta}_0 \: \text{(params)}; \: f(\bm{\theta}) \text{(objective)}; \: \gamma_t \:\text{(learning rate at } t \text{)}; \\
    &\hspace{17.25mm} \beta_1, \beta_2 \: \text{(betas)}; \: \lambda \: \text{(weight decay)}; \: \epsilon \text{ (epsilon)};\\
    &\hspace{17.25mm} \gamma_\text{max} \: \text{(maximum learning rate)}\\
    &\hspace{5mm} \text{initialize} : \bm{m}_{0} \leftarrow \bm{0}; \: \bm{v}_{0} \leftarrow \bm{0}\\[-0.5em]
    &\rule{105mm}{0.4pt}\\
    &\hspace{5mm} \textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}\text{:}\\
        &\hspace{10mm} \bm{g}_t \leftarrow \nabla_{\theta} f_t(\bm{\theta}_{t-1})\\[0.5em]
        &\hspace{10mm} \bm{m}_t \leftarrow \beta_1 \bm{m}_{t-1} + (1 - \beta_1) \bm{g}_t\\
        &\hspace{10mm} \bm{v}_t \leftarrow \beta_2 \bm{v}_{t-1} + (1 - \beta_2) \bm{g}^2_t\\[0.5em]
        &\hspace{10mm} \hat{\bm{m}}_t \leftarrow \bm{m}_t/(1 - \beta_1^t)\\
        &\hspace{10mm} \hat{\bm{v}}_t \leftarrow \bm{v}_t/(1 - \beta_2^t)\\[0.5em]
        &\hspace{10mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \gamma_t \bigl( \hat{\bm{m}}_t / (\sqrt{\hat{\bm{v}}_t} + \epsilon) \textcolor{#009ddb}{+ \lambda\bm{\theta}_{t-1}} \bigr)\textcolor{#9a3fe4}{- (\gamma_t/\gamma_\text{max})\lambda\bm{\theta}_{t-1}}\\[-0.5em]
    &\rule{105mm}{0.4pt}\\
\end{aligned}
$$

This difference applies to all optimi optimizers which implement both decoupled and fully decoupled weight decay.