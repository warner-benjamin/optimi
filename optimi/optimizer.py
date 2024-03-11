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
        if not 0.0 <= defaults["lr"]:
            raise ValueError(f"Invalid learning rate: lr={defaults['lr']}")
        if not 0.0 <= defaults["weight_decay"]:
            raise ValueError(f"Invalid weight decay: weight_decay={defaults['weight_decay']}")
        if defaults["decouple_lr"] and defaults["max_lr"] is None:
            defaults["max_lr"] = defaults["lr"]
        if defaults["max_lr"] is not None and not 0.0 <= defaults["max_lr"]:
            raise ValueError(f"Invalid maximum learning rate: max_lr={defaults['max_lr']}")

        if not MIN_TORCH_2_1:
            if defaults["foreach"]:
                raise ValueError(f"foreach={defaults['foreach']} requires PyTorch 2.1 or later. Set foreach=False or upgrade PyTorch.")
            else:
                defaults["foreach"] = False
            if defaults["gradient_release"]:
                raise ValueError(f"gradient_release={defaults['gradient_release']} requires PyTorch 2.1 or later. Upgrade PyTorch to use.")

        if defaults["decouple_lr"] and defaults["weight_decay"] >= 1e-3:
            warn(
                f"You are using weight_decay={defaults['weight_decay']} which is potentially high for decouple_lr={defaults['decouple_lr']}"
                f". Unlike decoupled weight decay, fully decoupled weight decay does not reduce weight decay by the learning rate.",
                category=UserWarning,
            )

        super().__init__(params, defaults)

        # by default perform the normal parameter update step
        self._optimizer_accumulation = False

        # if gradient_release is enabled, disable foreach step so normal optimizer step won't error
        if self.defaults["gradient_release"]:
            self.defaults["foreach"] = False
            for group in self.param_groups:
                group["foreach"] = False
                for p in group["params"]:
                    self.state[p]["group"] = group

    @property
    def optimizer_accumulation(self) -> bool:
        "Accumulate gradients in optimizer states during gradient release instead of a full step."
        return self._optimizer_accumulation

    @optimizer_accumulation.setter
    def optimizer_accumulation(self, optimizer_accumulation: bool):
        "Accumulate gradients in optimizer states during gradient release instead of a full step."
        self._optimizer_accumulation = optimizer_accumulation

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
