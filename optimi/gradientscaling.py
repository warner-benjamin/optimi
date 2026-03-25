from collections import defaultdict
import dataclasses

import torch
from torch import Tensor, optim


@dataclasses.dataclass(slots=True)
class ScalingInfo:
    """
    Args:
        gradient_scale: Current gradient scaling factor
        gradient_max_magnitude: Largest magnitude among all gradients (optional)
        optimizer_scale: Current optimizer scaling factor
        scale_down_hysteresis: When the optimizer scaling factor is updated downwards, then further divide the
            resulting scaling factor by this amount (larger than or equal to 1).
        scale_up_hysteresis: If not None, then the next step will automatically rescale the optimizer state to
            improve its coverage of the available dynamic range. This operation necessarily incurs a CPU-GPU
            sync, so you should not do it too often. If not None, then it must be a number greater than or equal
            to 1.
    """

    gradient_scale: float
    gradient_max_magnitude: float | None
    optimizer_scale: Tensor  # on CPU!
    scale_down_hysteresis: float
    scale_up_hysteresis: float | None

    def with_(self, **kw):
        return dataclasses.replace(self, **kw)


@dataclasses.dataclass
class MagnitudeLimits:
    gradient_max: dict[torch.dtype, float]
    optimizer_state_max: dict[torch.dtype, float]


DEFAULT_MAGNITUDE_LIMITS = MagnitudeLimits(gradient_max={torch.float16: 65536.0}, optimizer_state_max={torch.float16: 65504.0 * (7 / 8)})


def group_tensors_by_device_and_dtype(tensors: list[Tensor]) -> dict[tuple[torch.device, torch.dtype], list[Tensor]]:
    # lazy import to avoid affecting users who do not use gradient scaling
    from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

    if not tensors:
        return {}
    return {k: v for k, [[v], _] in _group_tensors_by_device_and_dtype([tensors]).items()}


@torch.no_grad()
def find_maximum_magnitude_single_device_and_dtype(tensors: list[Tensor], full: bool = False):
    lst = []
    for t in tensors:
        # Tensor.aminmax() propagates nans
        m = t.aminmax()
        lst.append(m.min)
        lst.append(m.max)

    if full:
        a = torch.stack(lst).abs().reshape(-1, 2).amax(dim=0)
        m = a.amax()
        return m, torch.isfinite(a)
    else:
        a = torch.stack(lst).abs().amax()
        return a


@torch.no_grad()
def find_maximum_upscale(
    tensors: list[Tensor], target_value_per_dtype: dict[torch.dtype, float], out_device="cpu"
) -> Tensor | list[Tensor]:
    """
    What is the greatest value `s` such that `(tensor * s).abs() <= target_value[tensor.dtype]` for every
    tensor in tensors?

    If `out_device` is None, then return a list of tensors belonging to different devices and dtypes. You are
    then responsible for computing the minimum across all of them.
    """
    results = [
        target_value / find_maximum_magnitude_single_device_and_dtype(ts)
        for (ts_device, ts_dtype), ts in group_tensors_by_device_and_dtype(tensors).items()
        if (target_value := target_value_per_dtype.get(ts_dtype)) is not None
    ]
    if out_device is None:
        return results
    else:
        if results:
            return torch.stack([s.to(device=out_device) for s in results]).amin()
        else:
            return torch.full((), 65536.0, dtype=torch.float32, device=out_device)


class _OptimizerScalingHelper:
    # used to define the types of non-dataclass fields
    gradient_max: float
    optimizer_max: float
    optimizer_scale: float
    gradient_to_optimizer_scale: float
    rescale_factor: float | None


@dataclasses.dataclass
class OptimizerScalingHelper(_OptimizerScalingHelper):
    """
    Args:
        scaling_info: Scaling information. This includes the gradient and optimizer scaling factors, and hysteresis
            factors to prevent overly frequent optimizer rescaling.

    Attributes:

        gradient_max: The maximum allowed input gradient value. This is set to be above the maximum representable
            float16 value, or provided by the gradient scaler.
        optimizer_max: The maximum allowed value inside the optimizer state. This is set just below the maximum
            representable float16 value. Input gradients that are at most `gradient_max` must not be able to
            produce an optimizer state that exceeds `optimizer_max`.
        optimizer_scale: The current scaling factor of the optimizer.
        gradient_to_optimizer_scale: The scaling factor by which to multiply a gradient value to obtain a value
            compatible with the current optimizer scaling.
        rescale_factor: If not None, then you MUST rescale the optimizer state by this factor on the next
            iteration.
    """

    scaling_info: ScalingInfo
    scale_opt_upper_limit_extra_factor: float = 1.0
    magnitude_limits = DEFAULT_MAGNITUDE_LIMITS

    def get_max_optimizer_state_scale_up_factor(self, dtype_to_max_value: dict[torch.dtype, float]) -> list[Tensor]:
        raise NotImplementedError

    def update(self):
        mag = self.magnitude_limits
        info = self.scaling_info

        # hard coded for now, sorry!
        gradient_max = mag.gradient_max[torch.float16]
        opt_state_max = mag.optimizer_state_max[torch.float16]

        if info.gradient_max_magnitude is not None:
            # Leave a tiny margin of safety.
            gradient_max = min(gradient_max, info.gradient_max_magnitude * 1.0625)

        self.gradient_max = gradient_max
        self.optimizer_max = opt_state_max

        # Compute upper limit on optimizer scale given the current gradient scaling factor (which may
        # have changed since the last iteration).
        scale_opt_upper_limit = self.scale_opt_upper_limit_extra_factor * opt_state_max / gradient_max * info.gradient_scale
        scale_opt = info.optimizer_scale.item()
        if scale_opt > scale_opt_upper_limit:
            # The current optimizer scale is larger than the upper allowed limit, so we must
            # scale the optimizer state downwards. This can happen when the gradients grow which results in
            # a shrinking `gradient_scale`.
            new_scale_opt = scale_opt_upper_limit / info.scale_down_hysteresis
        elif info.scale_up_hysteresis is not None:
            # User requested a periodic upscale, so we must figure out the maximum upscaling factor.

            # This returns a list of tensors that may or may not be on the CPU. We must take the minimum of
            # all of them.
            max_scale_up_opt_tensors = self.get_max_optimizer_state_scale_up_factor(
                dtype_to_max_value={k: v / info.scale_up_hysteresis for k, v in mag.optimizer_state_max.items()}
            )
            max_scale_up_opt = 65536.0
            if max_scale_up_opt_tensors:
                value = torch.stack([x.cpu() for x in max_scale_up_opt_tensors]).amin()
                if value.isfinite():
                    max_scale_up_opt = value.item()

            new_scale_opt = min(scale_opt * max_scale_up_opt, scale_opt_upper_limit)
            if new_scale_opt <= scale_opt:
                # this would not be an upscale, cancel it
                new_scale_opt = None
        else:
            new_scale_opt = None

        if new_scale_opt is not None:
            # A rescale is necessary.
            info.optimizer_scale.fill_(new_scale_opt)
            self.rescale_factor = new_scale_opt / scale_opt
            scale_opt = new_scale_opt
        else:
            self.rescale_factor = None

        self.optimizer_scale = scale_opt
        self.gradient_to_optimizer_scale = scale_opt / info.gradient_scale


class GradScalerBackport:
    """
    Backport of the newer PyTorch GradScaler.

    TODO: implement a better scaling policy
    """

    def __init__(
        self,
        device: str = "cpu",
        init_scale: float = 65536.0,
        *,
        growth_factor: float = 2.0 ** (1 / 50),
        backoff_factor: float = 0.5 ** (1 / 2),
        enabled: bool = True,
        gradient_max_magnitude: float = torch.finfo(torch.float16).max,
    ):
        self.device = device
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.enabled = enabled
        self.gradient_max_magnitude = gradient_max_magnitude
        self._found_inf = torch.full((), 0, dtype=torch.int32, device=device)
        self._scale = torch.full((), init_scale, dtype=torch.float32, device=device)
        self._gradient_max = torch.full((), 0.0, dtype=torch.float32, device=device)
        self._performed_step = False

    def scale(self, loss: Tensor | list[Tensor]):
        if not self.enabled:
            return loss

        if single := isinstance(loss, Tensor):
            loss = (loss,)

        loss = [x * self._scale.to(x.device) for x in loss]

        return loss[0] if single else loss

    def step(self, optimizer: optim.Optimizer, *args, **kwargs):
        if not self.enabled:
            return optimizer.step(*args, **kwargs)

        self._performed_step = True
        assert getattr(optimizer, "_step_supports_amp_scaling", False)

        self._update_found_inf(optimizer)
        optimizer.found_inf = self._found_inf
        optimizer.grad_scale = self._scale
        optimizer.grad_maximum_magnitude = self._gradient_max
        try:
            return optimizer.step(*args, **kwargs)
        finally:
            del optimizer.found_inf, optimizer.grad_scale, optimizer.grad_maximum_magnitude

    def _update_found_inf(self, optimizer):
        # adapted from torch.amp.GradScaler

        # https://stackoverflow.com/questions/5029934/defaultdict-of-defaultdict
        # Google says mypy struggles with defaultdicts type annotations.
        per_device_and_dtype_grads: dict[tuple[torch.device, torch.dtype], list[Tensor]] = defaultdict(list)
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    assert isinstance(param, torch.Tensor)
                    if param.grad is None:
                        continue
                    if param.grad.is_sparse:
                        # is_coalesced() == False means the sparse grad has values with duplicate indices.
                        # coalesce() deduplicates indices and adds all values that have the same index.
                        # For scaled fp16 values, there's a good chance coalescing will cause overflow,
                        # so we should check the coalesced _values().
                        if param.grad.dtype is torch.float16:
                            param.grad = param.grad.coalesce()
                        to_unscale = param.grad._values()
                    else:
                        to_unscale = param.grad

                    # TODO: is there a way to split by device and dtype without appending in the inner loop?
                    per_device_and_dtype_grads[to_unscale.device, to_unscale.dtype].append(to_unscale)

            maxs = []
            for k, grads in per_device_and_dtype_grads.items():
                m = find_maximum_magnitude_single_device_and_dtype(grads)
                maxs.append(m)

            device = self.device
            max_magnitude = torch.stack([m.to(device) for m in maxs]).amax()
            self._gradient_max.fill_(torch.maximum(self._gradient_max, max_magnitude))
            self._found_inf |= ~max_magnitude.isfinite() | (max_magnitude > self.gradient_max_magnitude)

    def get_scale(self):
        return self._scale.cpu().item()

    def update(self, new_scale=None):
        if not self._performed_step:
            self._found_inf.fill_(0)
            # nothing can be done
            return

        # CUDA-friendly, no CPU-GPU sync required if all values are on the same device.
        rescale = torch.where(self._found_inf != 0, self.backoff_factor, self.growth_factor)

        self._scale *= rescale
        self._found_inf.fill_(0)
        self._gradient_max.fill_(0.0)
        self._performed_step = False
