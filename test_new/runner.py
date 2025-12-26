from __future__ import annotations

import io
import random

import torch
from optimi import prepare_for_gradient_release, remove_gradient_release
from torch import Tensor

from .config import Backend, DeviceType, OptTest, OptTestType


def _device_type(device: torch.device) -> DeviceType:
    return DeviceType.cpu if device.type == "cpu" else DeviceType.gpu


def _get_iterations(
    opttest: OptTest,
    test_type: OptTestType,
    default: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> int:
    if not opttest.custom_iterations:
        return default
    if device is not None:
        key = (test_type, _device_type(device))
        if dtype is not None:
            dtype_key = (test_type, _device_type(device), dtype)
            if dtype_key in opttest.custom_iterations:
                return opttest.custom_iterations[dtype_key]
        if key in opttest.custom_iterations:
            return opttest.custom_iterations[key]
    return opttest.custom_iterations.get(test_type, default)


def assert_most_approx_close(
    a: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    max_error_count: int = 0,
    max_error_rate: float | None = None,
    name: str = "",
) -> None:
    """Assert that most values in two tensors are approximately close."""
    idx = torch.isclose(a.float(), b.float(), rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()

    if max_error_rate is not None:
        if error_count > (a.numel()) * max_error_rate and error_count > max_error_count:
            msg = f"{name}Too many values not close: assert {error_count} < {(a.numel()) * max_error_rate}"
            torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol, msg=msg)
    elif error_count > max_error_count:
        msg = f"{name}Too many values not close: assert {error_count} < {max_error_count}"
        torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol, msg=msg)


class MLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.norm = torch.nn.LayerNorm(input_size, device=device, dtype=dtype)
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act = torch.nn.Mish()
        self.fc2 = torch.nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def run_test(
    opttest: OptTest,
    device: torch.device,
    dtype: torch.dtype,
    backend: Backend,
    test_type: OptTestType,
    dims: tuple[int, int] | None = None,
) -> None:
    if test_type == OptTestType.normal:
        normal_spec = opttest.spec.normal
        normal_iters = normal_spec.iterations_cpu if device.type == "cpu" else normal_spec.iterations_gpu
        iterations = _get_iterations(opttest, test_type, normal_iters, device=device, dtype=dtype)
        tolerance = normal_spec.tolerance[dtype]

        if dims is None:
            dim1, dim2 = (64, 128) if device.type == "cpu" else (256, 512)
        else:
            dim1, dim2 = dims

        batch_size = normal_spec.batch_cpu if device.type == "cpu" else normal_spec.batch_gpu
        max_error_count = normal_spec.max_error_cpu if device.type == "cpu" else normal_spec.max_error_gpu
        max_error_rate = tolerance.max_error_rate

    elif test_type == OptTestType.gradient_release:
        gradient_spec = opttest.spec.gradient_release
        iterations = _get_iterations(opttest, test_type, gradient_spec.iterations, device=device, dtype=dtype)
        tolerance = gradient_spec.tolerance[dtype]

        dim1, dim2 = dims if dims is not None else (128, 256)
        batch_size = gradient_spec.batch
        max_error_count = gradient_spec.max_error_count
        max_error_rate = tolerance.max_error_rate

    elif test_type == OptTestType.accumulation:
        accumulation_spec = opttest.spec.accumulation
        iterations = _get_iterations(opttest, test_type, accumulation_spec.iterations, device=device, dtype=dtype)
        tolerance = accumulation_spec.tolerance[dtype]
        dim1, dim2 = dims if dims is not None else (128, 256)
        batch_size = accumulation_spec.batch
        max_error_count = 0
        max_error_rate = accumulation_spec.max_error_rate
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    m1 = MLP(dim1, dim2, device=device, dtype=dtype)
    m2 = MLP(dim1, dim2, device=device, dtype=dtype)
    m2.load_state_dict(m1.state_dict())

    if test_type == OptTestType.gradient_release:
        m3 = MLP(dim1, dim2, device=device, dtype=dtype)
        m3.load_state_dict(m1.state_dict())
    else:
        m3 = None

    if test_type == OptTestType.normal and not opttest.any_precision and dtype != torch.float32:
        for p in m1.parameters():
            p.data = p.data.float()

    reference_kwargs = opttest.to_reference_kwargs(backend)
    optimi_kwargs = opttest.to_optimi_kwargs(backend)
    reference_class = opttest.reference_class

    reference_optimizer = None
    optimi_optimizer = None
    torch_optimizers: dict[torch.nn.Parameter, torch.optim.Optimizer] | None = None
    pytorch_hooks: list[torch.utils.hooks.RemovableHandle] = []

    if test_type == OptTestType.normal:
        reference_optimizer = reference_class(m1.parameters(), **reference_kwargs)
        optimi_optimizer = opttest.optimi_class(m2.parameters(), **optimi_kwargs)
        buffer = io.BytesIO()
    elif test_type == OptTestType.gradient_release:
        reference_optimizer = reference_class(m1.parameters(), **reference_kwargs)

        def optimizer_hook(parameter) -> None:
            assert torch_optimizers is not None
            torch_optimizers[parameter].step()
            torch_optimizers[parameter].zero_grad()

        torch_optimizers = {p: reference_class([p], **reference_kwargs) for p in m2.parameters()}
        for p in m2.parameters():
            pytorch_hooks.append(p.register_post_accumulate_grad_hook(optimizer_hook))

        optimi_kwargs["gradient_release"] = True
        optimi_optimizer = opttest.optimi_class(m3.parameters(), **optimi_kwargs)
        prepare_for_gradient_release(m3, optimi_optimizer)
    else:
        reference_optimizer = reference_class(m1.parameters(), **reference_kwargs)
        optimi_kwargs["gradient_release"] = True
        optimi_optimizer = opttest.optimi_class(m2.parameters(), **optimi_kwargs)
        prepare_for_gradient_release(m2, optimi_optimizer)
        gradient_accumulation_steps = accumulation_spec.gradient_accumulation_steps

    for i in range(iterations):
        input1 = torch.randn(batch_size, dim1, device=device, dtype=dtype)
        if test_type == OptTestType.normal:
            input2 = input1.detach().clone()
        else:
            input2 = input1.clone()
        target1 = torch.randn(batch_size, 1, device=device, dtype=dtype)
        if test_type == OptTestType.normal:
            target2 = target1.detach().clone()
        else:
            target2 = target1.clone()

        if test_type == OptTestType.gradient_release:
            input3 = input1.clone()
            target3 = target1.clone()
        else:
            input3 = None
            target3 = None

        if test_type == OptTestType.normal and not opttest.any_precision and dtype != torch.float32:
            input1 = input1.float()
            target1 = target1.float()

        if test_type == OptTestType.accumulation:
            optimi_optimizer.optimizer_accumulation = (i + 1) % gradient_accumulation_steps != 0

        output1 = m1(input1)
        output2 = m2(input2)
        output3 = m3(input3) if m3 is not None else None

        loss1 = torch.nn.functional.mse_loss(output1, target1)
        loss2 = torch.nn.functional.mse_loss(output2, target2)
        loss3 = torch.nn.functional.mse_loss(output3, target3) if output3 is not None else None

        loss1.backward()
        loss2.backward()
        if loss3 is not None:
            loss3.backward()

        if test_type == OptTestType.normal:
            reference_optimizer.step()
            optimi_optimizer.step()
            reference_optimizer.zero_grad()
            optimi_optimizer.zero_grad()
        elif test_type == OptTestType.gradient_release:
            reference_optimizer.step()
            reference_optimizer.zero_grad()
        elif not optimi_optimizer.optimizer_accumulation:
            reference_optimizer.step()
            reference_optimizer.zero_grad()

        if test_type in (OptTestType.gradient_release, OptTestType.accumulation):
            if random.random() < 0.5:
                optimi_optimizer.step()
                optimi_optimizer.zero_grad()

        if test_type == OptTestType.normal:
            assert_most_approx_close(
                m1.fc1.weight,
                m2.fc1.weight,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
                max_error_count=max_error_count,
                max_error_rate=max_error_rate,
                name="fc1: ",
            )
            assert_most_approx_close(
                m1.fc2.weight,
                m2.fc2.weight,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
                max_error_count=max_error_count,
                max_error_rate=max_error_rate,
                name="fc2: ",
            )

            if i % max(1, iterations // 10) == 0 and i > 0:
                torch.save(optimi_optimizer.state_dict(), buffer)
                buffer.seek(0)
                ckpt = torch.load(buffer, weights_only=True)
                optimi_optimizer = opttest.optimi_class(m2.parameters(), **optimi_kwargs)
                optimi_optimizer.load_state_dict(ckpt)
                buffer.seek(0)
                buffer.truncate(0)

                assert_most_approx_close(
                    m1.fc1.weight,
                    m2.fc1.weight,
                    atol=tolerance.atol,
                    rtol=tolerance.rtol,
                    max_error_count=max_error_count,
                    max_error_rate=max_error_rate,
                    name="fc1 after load: ",
                )
                assert_most_approx_close(
                    m1.fc2.weight,
                    m2.fc2.weight,
                    atol=tolerance.atol,
                    rtol=tolerance.rtol,
                    max_error_count=max_error_count,
                    max_error_rate=max_error_rate,
                    name="fc2 after load: ",
                )
        elif test_type == OptTestType.gradient_release:
            assert_most_approx_close(
                m1.fc1.weight,
                m2.fc1.weight,
                rtol=tolerance.rtol,
                atol=tolerance.atol,
                max_error_count=max_error_count,
                max_error_rate=max_error_rate,
                name="PyTorch-PyTorch: ",
            )
            assert_most_approx_close(
                m1.fc2.weight,
                m2.fc2.weight,
                rtol=tolerance.rtol,
                atol=tolerance.atol,
                max_error_count=max_error_count,
                max_error_rate=max_error_rate,
                name="PyTorch-PyTorch: ",
            )
            assert_most_approx_close(
                m1.fc1.weight,
                m3.fc1.weight,
                rtol=tolerance.rtol,
                atol=tolerance.atol,
                max_error_count=max_error_count,
                max_error_rate=max_error_rate,
                name="PyTorch-Optimi: ",
            )
            assert_most_approx_close(
                m1.fc2.weight,
                m3.fc2.weight,
                rtol=tolerance.rtol,
                atol=tolerance.atol,
                max_error_count=max_error_count,
                max_error_rate=max_error_rate,
                name="PyTorch-Optimi: ",
            )

    if test_type == OptTestType.accumulation:
        assert_most_approx_close(
            m1.fc1.weight,
            m2.fc1.weight,
            rtol=tolerance.rtol,
            atol=tolerance.atol,
            max_error_count=max_error_count,
            max_error_rate=max_error_rate,
        )
        assert_most_approx_close(
            m1.fc2.weight,
            m2.fc2.weight,
            rtol=tolerance.rtol,
            atol=tolerance.atol,
            max_error_count=max_error_count,
            max_error_rate=max_error_rate,
        )

    for h in pytorch_hooks:
        h.remove()
    if test_type == OptTestType.gradient_release:
        remove_gradient_release(m3)
    elif test_type == OptTestType.accumulation:
        remove_gradient_release(m2)
