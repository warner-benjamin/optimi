---
title: "SGD: Stochastic Gradient Descent"
---

# SGD: Stochastic Gradient Descent

Stochastic Gradient Descent iteratively updates the model parameters using the gradient from a mini-batch of data.

SGD can be traced back to Herbert Robbins and Sutton Monro’s [stochastic approximation methods](https://doi.org/10.1214%2Faoms%2F1177729586). Frank Rosenblatt was the first to use SGD to train neural networks in [*The perceptron: A probabilistic model for information storage and organization in the brain*](https://doi.org/10.1037%2Fh0042519).

## Hyperparmeters

SGD supports dampening `dampening=True`, where `dampening=1-momentum`. To match PyTorch’s dampening set `torch_init=True`. This will initialize momentum buffer with first gradient instead of zeroes.

optimi’s implementation of SGD also supports decoupled weight decay `decouple_wd=True` and [fully decoupled weight decay](../fully_decoupled_weight_decay.md) `decouple_lr=True`. Weight decay will likely [need to be reduced](../fully_decoupled_weight_decay.md#hyperparameters) when using fully decoupled weight decay as the learning rate will not modify the effective weight decay.

::: optimi.sgd.SGD

## Algorithm

SGD with L2 regularization.

$$
\begin{aligned}
    &\rule{100mm}{0.4pt}\\
    &\hspace{2mm} \textcolor{#dc3918}{\textbf{SGD}} \: \textcolor{#009ddb}{\text{with momentum}} \: \textcolor{#9a3fe4}{\text{and dampening}}\\
    &\hspace{5mm} \text{inputs} : \bm{\theta}_0 \: \text{(params)}; \: f(\bm{\theta}) \text{(objective)}; \: \gamma_t \:\text{(learning rate at } t \text{)}; \\
    &\hspace{17.25mm} \beta \: \text{(momentum)}; \: \lambda \: \text{(weight decay)}\\
    &\hspace{5mm} \text{initialize} : \textcolor{#009ddb}{\bm{m}_{0} \leftarrow \bm{0}}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
    &\hspace{5mm} \textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}\text{:}\\
        &\hspace{10mm} \bm{g}_t \leftarrow \nabla_{\theta} f_t(\bm{\theta}_{t-1}) - \lambda\bm{\theta}_{t-1}\\
        &\hspace{10mm} \textcolor{#009ddb}{\bm{m}_t \leftarrow \beta \bm{m}_{t-1} +} \textcolor{#9a3fe4}{(1 - \beta)} \textcolor{#009ddb}{\bm{g}_t}\\
        &\hspace{10mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} \textcolor{#dc3918}{- \gamma_t\bm{g}_t} \textcolor{#009ddb}{- \gamma_t\bm{m}_t}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
\end{aligned}
$$

The SGD update terms $\gamma_t\bm{g}_t$ and $\gamma_t\bm{m}_t$ are exclusive, applying for SGD and SGD with momentum (and dampening), respectively. The dampening term $(1 - \beta)$ is added to the momentum update $\bm{m}_t \leftarrow \beta \bm{m}_{t-1} + \bm{g}_t$ if enabled.

optimi’s SGD also supports [AdamW’s](adamw.md#algorithm) decoupled weight decay and [fully decoupled weight decay](../fully_decoupled_weight_decay.md#algorithm), which are not shown.