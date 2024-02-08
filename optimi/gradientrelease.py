from __future__ import annotations

from functools import partial
from warnings import warn

from torch import Tensor
from torch.nn import Module

from optimi.optimizer import OptimiOptimizer


def _gradient_release_hook(param: Tensor, optimizer: OptimiOptimizer):
    optimizer.step(param=param)
    optimizer.zero_grad(param=param)


def prepare_for_gradient_release(model: Module, optimizer: OptimiOptimizer, ignore_existing_hooks: bool = False):
    """Register post_accumulate_grad_hooks on parameters for the gradient release optimization step.

    Args:
        model: Model to register post_accumulate_grad_hooks. Only registers on parameters with
            `requires_grad=True`.
        optimizer: Optimizer providing the fused optimizer step during the backward pass. Requires
            optimizer to be initialized with `gradient_release=True`
        ignore_existing_hooks: If True, ignores existing post_accumulate_grad_hooks on parameters
            and registers gradient release hooks (default: False)
    """
    if not isinstance(optimizer, OptimiOptimizer):
        raise TypeError("`optimizer` must be an instance of `OptimiOptimizer`")
    if not optimizer.defaults["gradient_release"]:
        raise ValueError("`optimizer` must be initialized with `gradient_release=True`")

    hooks = []
    for p in model.parameters():
        if p.requires_grad:
            if (p._post_accumulate_grad_hooks is not None) and len(p._post_accumulate_grad_hooks) > 0 and (not ignore_existing_hooks):
                for hook in hooks:
                    if hasattr(hook, "remove"):
                        hook.remove()
                raise ValueError(
                    "Model already has post_accumulate_grad_hooks. If this is expected, rerun with `ignore_existing_hooks=True`."
                )
            hooks.append(p.register_post_accumulate_grad_hook(partial(_gradient_release_hook, optimizer=optimizer)))
    model._gradient_release_hooks = hooks


def remove_gradient_release(model: Module):
    """Removes post_accumulate_grad_hooks created by `prepare_for_gradient_release`.

    Args:
        model: Model to remove gradient release post_accumulate_grad_hooks from.
    """
    if not hasattr(model, "_gradient_release_hooks"):
        warn("`model` does not have any gradient release post_accumulate_grad_hooks to remove.")
        return

    for hook in model._gradient_release_hooks:
        if hasattr(hook, "remove"):
            hook.remove()
    del model._gradient_release_hooks
