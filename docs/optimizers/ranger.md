---
title: "Ranger: RAdam and LookAhead"
---

# Ranger: RAdam and LookAhead

Ranger combines [RAdam](radam.md) and [Lookahead](https://arxiv.org/abs/1907.08610) together in one optimizer. RAdam fixes the adaptive learning rate's large variance during early stages of training to improve convergence and reducing the need for warmup. Lookahead updates model weights like normal, but every k steps interpolates them with a copy of slow moving weights. This moving average of the model weights is less sensitive to suboptimal hyperparameters and reduces the need for hyperparameter tuning.

Ranger was introduced by Less Wright in [*New Deep Learning Optimizer, Ranger: Synergistic combination of RAdam + Lookahead for the best of both*](https://lessw.medium.com/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d).

## Hyperparameters

Ranger works best with a flat learning rate followed by a short learning rate decay. Try raising the learning rate 2-3x larger than [AdamW](adamw.md).

optimi sets the default $\beta$s to `(0.9, 0.99)` and default $\epsilon$ to `1e-6`. These values reflect current best-practices and usually outperform the PyTorch defaults.

optimi’s implementation of Ranger supports both [decoupled weight decay](adamw.md) `decouple_wd=True` and [fully decoupled weight decay](../fully_decoupled_weight_decay.md) `decouple_lr=True`. Weight decay will likely [need to be reduced](../fully_decoupled_weight_decay.md#hyperparameters) when using fully decoupled weight decay as the learning rate will not modify the effective weight decay.

::: optimi.ranger.Ranger

## Algorithm

Ranger: RAdam and LookAhead.

$$
\begin{aligned}
    &\rule{100mm}{0.4pt}\\
    &\hspace{2mm} \textbf{Ranger: RAdam and} \: \textcolor{#9a3fe4}{\textbf{LookAhead}}\\
    &\hspace{5mm} \text{inputs} : \bm{\theta}_0 \: \text{(params)}; \: f(\bm{\theta}) \text{(objective)}; \: \gamma_t \:\text{(learning rate at } t \text{)}; \\
    &\hspace{17.25mm} \beta_1, \beta_2 \: \text{(betas)}; \: \lambda \: \text{(weight decay)}; \: \epsilon \: \text{(epsilon)};\\
    &\hspace{17.25mm} \bm{\phi}_0 \: \text{(slow params)}; \: k \: \text{(sync)}; \: \alpha \: \text{(interpolation)};\\
    &\hspace{5mm} \text{initialize} : \textcolor{#9a3fe4}{\bm{\phi}_0 \leftarrow  \bm{\theta}_0}; \: \bm{m}_{0} \leftarrow \bm{0}; \: \bm{v}_{0} \leftarrow \bm{0}; \: \rho_{\infty} \leftarrow 2 / (1 - \beta_2) - 1;\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
    &\hspace{5mm} \textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}\text{:}\\
        &\hspace{10mm} \bm{g}_t \leftarrow \nabla_{\theta} f_t(\bm{\theta}_{t-1})\\[0.5em]
        &\hspace{10mm} \bm{m}_t \leftarrow \beta_1 \bm{m}_{t-1} + (1 - \beta_1) \bm{g}_t\\
        &\hspace{10mm} \bm{v}_t \leftarrow \beta_2 \bm{v}_{t-1} + (1 - \beta_2) \bm{g}^2_t\\[0.5em]
        &\hspace{10mm} \hat{\bm{m}}_t \leftarrow \bm{m}_t/(1 - \beta_1^t)\\
        &\hspace{10mm} \hat{\bm{v}}_t \leftarrow \bm{v}_t/(1 - \beta_2^t)\\[0.5em]
        &\hspace{10mm} \rho_t \leftarrow \rho_{\infty} - 2 t \beta^t_2 /(1 - \beta_2^t)\\[0.5em]
        &\hspace{10mm} \textbf{if} \: \rho_t > 5\text{:}\\
        &\hspace{15mm} r_t \leftarrow \sqrt{\tfrac{(\rho_t - 4)(\rho_t - 2)\rho_{\infty}}{(\rho_{\infty} - 4)(\rho_{\infty} -2 ) \rho_t}}\\
        &\hspace{15mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \gamma_t r_t \bigl( \hat{\bm{m}}_t / (\sqrt{\hat{\bm{v}}_t} + \epsilon) + \lambda\bm{\theta}_{t-1} \bigr)\\
        &\hspace{10mm} \textbf{else}\text{:}\\
        &\hspace{15mm} \bm{\theta}_t \leftarrow \bm{\theta}_{t-1} - \gamma_t (\hat{\bm{m}}_t + \lambda\bm{\theta}_{t-1})\\[0.5em]
        &\hspace{10mm} \textcolor{#9a3fe4}{\textbf{if} \: t \equiv 0 \pmod{k}\text{:}}\\
        &\hspace{15mm} \textcolor{#9a3fe4}{\bm{\phi}_t \leftarrow \bm{\phi}_{t-k} + \alpha(\bm{\theta}_t - \bm{\phi}_{t-k} )}\\
        &\hspace{15mm} \textcolor{#9a3fe4}{\bm{\theta}_t  \leftarrow \bm{\phi}_t}\\[-0.5em]
    &\rule{100mm}{0.4pt}\\
\end{aligned}
$$

optimi’s Ranger also supports L2 regularization and [fully decoupled weight decay](../fully_decoupled_weight_decay.md#algorithm), which are not shown.