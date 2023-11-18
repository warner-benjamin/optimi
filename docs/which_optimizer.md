---
title: Which Optimizer Should I Use?
---

# Which Optimizer Should I Use?

This guide is meant to provide a quick overview for choosing an optimi optimizer.

All optimi optimizers support training in pure BFloat16 precision[^1] using [Kahan summation](kahan_summation.md), which can help match Float32 optimizer performance with  [mixed precision](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch) while [reducing memory usage](kahan_summation.md#memory-savings) and [increasing training speed](kahan_summation.md#training-speedup).

## Tried and True

There’s a reason [AdamW](optimizers/adamw.md) is the default optimizer of deep learning. It performs well across multiple domains, model architectures, and batch sizes. Most optimizers claiming to outperform AdamW usually do not after careful analysis and experimentation.

Consider [reducing](optimizers/adamw.md#hyperparameters) the $\beta_2$ term if training on large batch sizes or observing training loss spikes[^2].

## Drop-in Replacement

If using gradient clipping during training or experience training loss spikes, try replacing AdamW with [StableAdamW](optimizers/stableadamw.md). StableAdamW applies AdaFactor style update clipping to AdamW, stabilizing training loss and removing the need for gradient clipping.

StableAdamW can outperform AdamW with gradient clipping on downstream tasks.

## Low Memory Usage

If optimizer memory usage is important and optimi’s Kahan summation doesn’t alleviate optimizer memory usage or even more memory savings are desired, try optimi’s two low memory optimizers: [Lion](optimizers/lion.md) and [SGD](optimizers/sgd.md).

Lion uses one memory buffer for both momentum and the update step, reducing memory usage compared to AdamW. While [reviews are mixed](https://arxiv.org/abs/2307.06440), Lion can [match AdamW in some training scenarios](https://github.com/lucidrains/lion-pytorch/discussions/1).

Prior to Adam and AdamW, SGD was the default optimizer for deep learning. SGD with Momentum can match or outperform AdamW on some tasks but can require more hyperparameter tuning. Consider using SGD with decoupled weight decay, it can lead to better results than L2 regularization.

## Potential Upgrade

[Adan](optimizers/adan.md) can outperform AdamW at the expense of extra memory usage due to using two more buffers then AdamW. Consider trying Adan if optimizer memory usage isn’t a priority, or when finetuning.


[^1]: Or BFloat16 with normalization and RoPE layers in Float32.

[^2]: This setting is [mentioned](https://twitter.com/giffmana/status/1692641748445438301) in [*Sigmoid Loss for Language Image Pre-Training*](https://arxiv.org/abs/2303.15343), although it is common knowledge in parts of the deep learning community.