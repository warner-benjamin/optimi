from __future__ import annotations

from typing import Any, Callable, Iterable
from warnings import warn

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from optimi.utils import MIN_TORCH_2_1


class OptimiOptimizer(Optimizer):
    """Provides common functionality for optimi optimizers."""

    def __init__(self, params: Iterable[Tensor] | Iterable[dict], defaults: dict[str, Any]):
        if not MIN_TORCH_2_1:
            if defaults["foreach"]:
                foreach = defaults["foreach"]
                raise ValueError(f"{foreach=} requires PyTorch 2.1 or later. Set foreach=False or upgrade PyTorch.")
            else:
                defaults["foreach"] = False

        if defaults["decouple_lr"] and defaults["weight_decay"] >= 1e-3:
            weight_decay = defaults["weight_decay"]
            decouple_lr = defaults["decouple_lr"]
            warn(
                f"You are using {weight_decay=} which is potentially high for {decouple_lr=}. Unlike decoupled weight "
                f"decay, fully decoupled weight decay does not reduce weight decay by the learning rate.",
                category=UserWarning,
            )

        super().__init__(params, defaults)

        if self.defaults["gradient_release"]:
            for group in self.param_groups:
                for p in group["params"]:
                    self.state[p]["group"] = group

    def step(self, closure: Callable | None = None, param: Tensor | None = None):
        """Performs a single optimization step on the whole model or individual parameter.

        Args:
            closure: A closure which reevaluates the model and returns the loss. Incompatible with
                performing an optimization step on a single `param`.
            param: An individual parameter to perform a fused optimization step during the backward
                pass. Requires optimizer to be initialized with `gradient_release=True` and model
                hooks created with `register_gradient_release`. Incompatible with `closure`.
        """
        raise NotImplementedError

    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True, param: Tensor | None = None):
        """Resets the gradients of all optimized parameters or individual parameter.

        Args:
            set_to_none: If True, the gradients will be deallocated after the call (default: True)
            param: Resets the gradients of the passed `param`. For use with `gradient_release=True`.
        """
        if param is None:
            super().zero_grad(set_to_none=set_to_none)
        else:
            if param.grad is not None:
                if set_to_none:
                    param.grad = None
                else:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
