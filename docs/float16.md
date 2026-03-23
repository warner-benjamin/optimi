# Float16 training

## Motivation

Pure float16 training is tricky due to the limited dynamic range of float16 numbers. The largest representable number is 65504 (approximately $2^{16}$), and the smallest positive normal number is $6.103\times 10^{-5}$ (approximately $2^{-14}$). This is in stark contrast with float32 and bfloat16 which both share the same dynamic range that is able to represent numbers as large as $10^{38}$ and as small as $10^-{38}$. The most immediate problem with pure float16 training is that the loss function is usually around 1 in magnitude, so the gradients ($\partial L/\partial w_i$) end up on the order of $10^{-3}$ to $10^{-7}$. As a result, many gradients underflow or become [subnormal](https://en.wikipedia.org/wiki/Subnormal_number) and lose precision. Usually, the model simply fails to converge.

When a machine learning practitioner encounters this problem, the solution suggested to them is [gradient scaling](https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling) which scales up the loss (and therefore the gradients) before backpropagation, and then un-scales these gradients inside the optimizer right before using them. However, the optimizer state usually still uses the same low-bit dtype (i.e., float16) to store values that are derived from the gradients and which have similarly tiny magnitudes. Even worse, some state values are proportional to the square of the gradients (Adam's exponential moving average of the gradient variance for example). Squaring a gradient with magnitude $10^{-5}$ results in a variance value of $10^{-10}$, laughably far below float16's limited dynamic range.

The next obvious solution is to scale up the loss (and gradients) and then simply not downscale them inside the optimizer. This certainly prevents the loss of precision when unscaling gradients. It works well with scaling factors that bring the magnitude of the gradients to be around 1, and it is an improvement over not scaling the gradients at all. It may be tempting to scale the gradients such that they cover the full dynamic range of float16 values. But consider a (scaled) gradient value of 1024. Inside Adam, the corresponding variance is its square $(2^{10})^2 = 2^{20}$, far above the maximum representable float16 number (65504), thus causing an overflow.

What we want is to ensure that all of the state variables take full advantage of float16's limited dynamic range. The different state variables must then be scaled by different amounts. Each state variable would introduce a new scaling factor to tune, and each type of optimizer (SGD, RMSProp, Adam, etc) would have its own set of new hyperparameters. This is burdensome to the user. Is there a way to unify these scaling parameters?

Yes - we can assume that the gradient is within a certain range, and automatically choose state variable scaling factors that do not overflow if the gradient is within the range. That is, **if**

- The magnitudes of the scaled gradients are less than or equal to `gradient_max`.
- The magnitudes of the scaled old optimizer state variables are less than or equal to `optimizer_state_max`.

**then**

- The magnitude of the new scaled optimizer state variables will be less than or equal to `optimizer_state_max`.
