---
title: "Lion: Evolved Sign Momentum"
---

# Lion: Evo**L**ved S**i**gn M**o**me**n**tum

Lion only keeps track of the gradient moving average (momentum) which can reduce memory usage compared to [AdamW](adamw.md). Lion uses two momentum EMA factors, one for tracking momentum and another for using momentum in the update step. Using default hyperparameters, this allows up to ten times longer history for momentum tracking while leveraging more of the current gradient for the model update. Unlike most optimizers, Lion uses the same magnitude for each parameter update calculated using the sign operation.

Lion was introduced by Chen et al in *[Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)*.

## Hyperparameters

Hyperparameter notes from Chen et al:

1. Due to the larger update norm from the sign operation, a good Lion learning rate is typically 3-10X smaller than [AdamW](adamw.md).
2. Since the effective weight decay is multiplied by the learning rate[^1], weight decay should be increased by the learning rate decrease (3-10X).
3. Except for language modeling, $\beta$s are set to `(0.9, 0.99)`. When training T5, Chen at al set $\beta_1=0.95$ and $\beta_2=0.98$. Reducing $\beta_2$ results in better training stability due to less historical memorization.
4. The optimal batch size for Lion is 4096 (vs AdamW’s 256), but Lion still performs well at a batch size of 64 and matches or exceeds AdamW on all tested batch sizes.

optimi’s implementation of Lion also supports [fully decoupled weight decay](../fully_decoupled_weight_decay.md) `decouple_lr=True`. If using fully decoupled weight decay do not increase weight decay. Rather, weight decay will likely [need to be reduced](../fully_decoupled_weight_decay.md#hyperparameters) as the learning rate will not modify the effective weight decay.

::: optimi.lion.Lion

## Algorithm

Lion: Evolved Sign Momentum.

$$
\begin{aligned}
    &\rule{100mm}{0.4pt}\\
    &\hspace{2mm} \textbf{Lion} \\
    &\hspace{5mm} \text{inputs} : \bm{\theta}_0 \: \text{(params)}; \: f(\bm{\theta}) \text{(objective)}; \: \gamma_t \:\text{(learning rate at } t \text{)}; \\
    &\hspace{17.25mm} \beta_1, \beta_2 \: \text{(betas)}; \: \lambda \: \text{(weight decay)}\\
    &\hspace{5mm} \text{initialize} : \bm{m}_{0} \leftarrow \bm{0}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
    &\hspace{5mm} \textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}\text{:}\\
        &\hspace{10mm} \bm{g}_t \leftarrow \nabla_{\theta} f_t(\bm{\theta}_{t-1})\\[0.5em]
        &\hspace{10mm} \bm{u} \leftarrow \beta_1 \bm{m}_{t-1} + (1 - \beta_1) \bm{g}_t\\
        &\hspace{10mm} \bm{m}_t \leftarrow \beta_2 \bm{m}_{t-1} + (1 - \beta_2) \bm{g}_t\\[0.5em]
        &\hspace{10mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \gamma_t \bigl(\text{sign}(\bm{u}) + \lambda\bm{\theta}_{t-1} \bigr)\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
\end{aligned}
$$

optimi’s Lion also supports [fully decoupled weight decay](../fully_decoupled_weight_decay.md#algorithm), which is not shown.

[^1]: The learning rate does not modify the effective weight decay when using fully decoupled weight decay.