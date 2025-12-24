from __future__ import annotations

import io
import random

import torch
from optimi import prepare_for_gradient_release, remove_gradient_release
from torch import Tensor

from .cases import Backend, DeviceType, OptTest, TestType, Tolerance
from .config import DEFAULTS


def _device_type(device: torch.device) -> DeviceType:
    return DeviceType.cpu if device.type == "cpu" else DeviceType.gpu


def _get_iterations(
    case: OptTest,
    test_type: TestType,
    default: int,
    device: torch.device | None = None,
) -> int:
    if not case.custom_iterations:
        return default
    if device is not None:
        key = (test_type, _device_type(device))
        if key in case.custom_iterations:
            return case.custom_iterations[key]
    return case.custom_iterations.get(test_type, default)


def assert_most_approx_close(
    a: torch.Tensor,
    b: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    max_error_count: int = 0,
    max_error_rate: float | None = None,
    name: str = "",
) -> None:
    """Assert that most values in two tensors are approximately close.

    Allows for a small number of errors based on max_error_count and max_error_rate.
    """
    idx = torch.isclose(a.float(), b.float(), rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()

    if max_error_rate is not None:
        if error_count > (a.numel()) * max_error_rate and error_count > max_error_count:
            print(f"{name}Too many values not close: assert {error_count} < {(a.numel()) * max_error_rate}")
            torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol)
    elif error_count > max_error_count:
        print(f"{name}Too many values not close: assert {error_count} < {max_error_count}")
        torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol)


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


def run_correctness(
    case: OptTest,
    device: torch.device,
    dtype: torch.dtype,
    backend: Backend,
    dims: tuple[int, int] | None = None,
) -> None:
    # Iterations and tolerance
    default_iters = DEFAULTS.correctness.cpu_iterations if device.type == "cpu" else DEFAULTS.correctness.gpu_iterations
    iterations = _get_iterations(case, TestType.correctness, default_iters, device=device)
    # Special-case: Adan bf16 on GPU
    if device.type != "cpu" and dtype == torch.bfloat16 and case.optimizer_name == "adan":
        iterations = DEFAULTS.correctness.adan_bf16_gpu_iterations
    tolerance = case.custom_tolerances[dtype]
    # Dims, batch, errors
    if dims is not None:
        dim1, dim2 = dims
    elif device.type == "cpu":
        dim1, dim2 = DEFAULTS.correctness.cpu_dims
    else:
        dim1, dim2 = DEFAULTS.correctness.gpu_dims
    batch_size = DEFAULTS.correctness.cpu_batch_size if device.type == "cpu" else DEFAULTS.correctness.gpu_batch_size
    max_error_count = DEFAULTS.correctness.cpu_max_error_count if device.type == "cpu" else DEFAULTS.correctness.gpu_max_error_count

    # Create models
    m1 = MLP(dim1, dim2, device=device, dtype=dtype)
    m2 = MLP(dim1, dim2, device=device, dtype=dtype)
    m2.load_state_dict(m1.state_dict())

    # Convert parameters to float for non-any_precision
    if not case.any_precision and dtype != torch.float32:
        for p in m1.parameters():
            p.data = p.data.float()

    # Optimizers
    reference_class = case.reference_class
    reference_kwargs = case.to_reference_kwargs(backend)
    optimi_kwargs = case.to_optimi_kwargs(backend)
    reference_optimizer = reference_class(m1.parameters(), **reference_kwargs)
    optimi_optimizer = case.optimi_class(m2.parameters(), **optimi_kwargs)

    buffer = io.BytesIO()

    for i in range(iterations):
        input1 = torch.randn(batch_size, dim1, device=device, dtype=dtype)
        input2 = input1.detach().clone()
        target1 = torch.randn(batch_size, 1, device=device, dtype=dtype)
        target2 = target1.detach().clone()

        if not case.any_precision and dtype != torch.float32:
            input1 = input1.float()
            target1 = target1.float()

        output1 = m1(input1)
        output2 = m2(input2)
        loss1 = torch.nn.functional.mse_loss(output1, target1)
        loss2 = torch.nn.functional.mse_loss(output2, target2)
        loss1.backward()
        loss2.backward()

        reference_optimizer.step()
        optimi_optimizer.step()
        reference_optimizer.zero_grad()
        optimi_optimizer.zero_grad()

        assert_most_approx_close(
            m1.fc1.weight,
            m2.fc1.weight,
            atol=tolerance.atol,
            rtol=tolerance.rtol,
            max_error_count=max_error_count,
            max_error_rate=tolerance.max_error_rate,
            name="fc1: ",
        )
        assert_most_approx_close(
            m1.fc2.weight,
            m2.fc2.weight,
            atol=tolerance.atol,
            rtol=tolerance.rtol,
            max_error_count=max_error_count,
            max_error_rate=tolerance.max_error_rate,
            name="fc2: ",
        )

        # state_dict save/load periodically
        if i % max(1, iterations // 10) == 0 and i > 0:
            torch.save(optimi_optimizer.state_dict(), buffer)
            buffer.seek(0)
            ckpt = torch.load(buffer, weights_only=True)
            optimi_optimizer = case.optimi_class(m2.parameters(), **optimi_kwargs)
            optimi_optimizer.load_state_dict(ckpt)
            buffer.seek(0)
            buffer.truncate(0)

            assert_most_approx_close(
                m1.fc1.weight,
                m2.fc1.weight,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
                max_error_count=max_error_count,
                max_error_rate=tolerance.max_error_rate,
                name="fc1 after load: ",
            )
            assert_most_approx_close(
                m1.fc2.weight,
                m2.fc2.weight,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
                max_error_count=max_error_count,
                max_error_rate=tolerance.max_error_rate,
                name="fc2 after load: ",
            )


def run_gradient_release(
    case: OptTest,
    device: torch.device,
    dtype: torch.dtype,
    backend: Backend,
    dims: tuple[int, int] | None = None,
) -> None:
    def optimizer_hook(parameter) -> None:
        torch_optimizers[parameter].step()
        torch_optimizers[parameter].zero_grad()

    # Iterations
    iterations = _get_iterations(case, TestType.gradient_release, DEFAULTS.gradient_release.iterations, device=device)

    # Tolerances: merge baseline with per-case
    tol = case.custom_tolerances[dtype]
    baseline = DEFAULTS.gradient_release.baseline_tolerance.get(dtype, tol)
    tolerance = Tolerance(
        rtol=max(tol.rtol, baseline.rtol),
        atol=max(tol.atol, baseline.atol),
        max_error_rate=max(tol.max_error_rate, baseline.max_error_rate),
        equal_nan=tol.equal_nan,
    )

    max_error_count = DEFAULTS.gradient_release.max_error_count

    # Dims and batch size
    if dims is not None:
        dim1, dim2 = dims
    else:
        dim1, dim2 = DEFAULTS.gradient_release.dims

    batch_size = DEFAULTS.gradient_release.batch_size

    m1 = MLP(dim1, dim2, device=device, dtype=dtype)  # regular
    m2 = MLP(dim1, dim2, device=device, dtype=dtype)  # PyTorch hooks
    m3 = MLP(dim1, dim2, device=device, dtype=dtype)  # Optimi gradient release
    m2.load_state_dict(m1.state_dict())
    m3.load_state_dict(m1.state_dict())

    reference_class = case.reference_class
    reference_kwargs = case.to_reference_kwargs(backend)
    optimi_kwargs = case.to_optimi_kwargs(backend)

    regular_optimizer = reference_class(m1.parameters(), **reference_kwargs)
    torch_optimizers = {p: reference_class([p], **reference_kwargs) for p in m2.parameters()}
    pytorch_hooks = []
    for p in m2.parameters():
        pytorch_hooks.append(p.register_post_accumulate_grad_hook(optimizer_hook))

    optimi_kwargs["gradient_release"] = True
    optimi_optimizer = case.optimi_class(m3.parameters(), **optimi_kwargs)
    prepare_for_gradient_release(m3, optimi_optimizer)

    for _ in range(iterations):
        input1 = torch.randn(batch_size, dim1, device=device, dtype=dtype)
        input2 = input1.clone()
        input3 = input1.clone()
        target1 = torch.randn(batch_size, 1, device=device, dtype=dtype)
        target2 = target1.clone()
        target3 = target1.clone()

        output1 = m1(input1)
        output2 = m2(input2)
        output3 = m3(input3)

        loss1 = torch.nn.functional.mse_loss(output1, target1)
        loss2 = torch.nn.functional.mse_loss(output2, target2)
        loss3 = torch.nn.functional.mse_loss(output3, target3)

        loss1.backward()
        loss2.backward()
        loss3.backward()

        regular_optimizer.step()
        regular_optimizer.zero_grad()

        # Random step/zero_grad to simulate using optimi's accumulation in a framework like Composer
        if random.random() < 0.5:
            optimi_optimizer.step()
            optimi_optimizer.zero_grad()

        assert_most_approx_close(
            m1.fc1.weight,
            m2.fc1.weight,
            rtol=tolerance.rtol,
            atol=tolerance.atol,
            max_error_count=max_error_count,
            max_error_rate=tolerance.max_error_rate,
            name="PyTorch-PyTorch: ",
        )
        assert_most_approx_close(
            m1.fc2.weight,
            m2.fc2.weight,
            rtol=tolerance.rtol,
            atol=tolerance.atol,
            max_error_count=max_error_count,
            max_error_rate=tolerance.max_error_rate,
            name="PyTorch-PyTorch: ",
        )
        assert_most_approx_close(
            m1.fc1.weight,
            m3.fc1.weight,
            rtol=tolerance.rtol,
            atol=tolerance.atol,
            max_error_count=max_error_count,
            max_error_rate=tolerance.max_error_rate,
            name="PyTorch-Optimi: ",
        )
        assert_most_approx_close(
            m1.fc2.weight,
            m3.fc2.weight,
            rtol=tolerance.rtol,
            atol=tolerance.atol,
            max_error_count=max_error_count,
            max_error_rate=tolerance.max_error_rate,
            name="PyTorch-Optimi: ",
        )

    for h in pytorch_hooks:
        h.remove()
    remove_gradient_release(m3)


def run_accumulation(
    case: OptTest,
    device: torch.device,
    dtype: torch.dtype,
    backend: Backend,
    dims: tuple[int, int] | None = None,
) -> None:
    # Iterations
    iterations = _get_iterations(case, TestType.accumulation, DEFAULTS.accumulation.iterations, device=device)

    # Dims and batch size
    if dims is not None:
        dim1, dim2 = dims
    else:
        dim1, dim2 = DEFAULTS.accumulation.dims

    batch_size = DEFAULTS.accumulation.batch_size

    # Tolerance and error rate
    tolerance = DEFAULTS.accumulation.tolerance
    max_error_rate = DEFAULTS.accumulation.max_error_rate

    m1 = MLP(dim1, dim2, device=device, dtype=dtype)  # Regular optimizer
    m2 = MLP(dim1, dim2, device=device, dtype=dtype)  # Optimi accumulation
    m2.load_state_dict(m1.state_dict())

    reference_class = case.reference_class
    reference_kwargs = case.to_reference_kwargs(backend)
    optimi_kwargs = case.to_optimi_kwargs(backend)

    regular_optimizer = reference_class(m1.parameters(), **reference_kwargs)
    optimi_kwargs["gradient_release"] = True
    optimi_optimizer = case.optimi_class(m2.parameters(), **optimi_kwargs)
    prepare_for_gradient_release(m2, optimi_optimizer)

    gradient_accumulation_steps = DEFAULTS.accumulation.gradient_accumulation_steps

    for i in range(iterations):
        input1 = torch.randn(batch_size, dim1, device=device, dtype=dtype)
        input2 = input1.clone()
        target1 = torch.randn(batch_size, 1, device=device, dtype=dtype)
        target2 = target1.clone()

        optimi_optimizer.optimizer_accumulation = (i + 1) % gradient_accumulation_steps != 0

        output1 = m1(input1)
        output2 = m2(input2)
        loss1 = torch.nn.functional.mse_loss(output1, target1)
        loss2 = torch.nn.functional.mse_loss(output2, target2)

        loss1.backward()
        loss2.backward()

        if not optimi_optimizer.optimizer_accumulation:
            regular_optimizer.step()
            regular_optimizer.zero_grad()

        # Random step/zero_grad to simulate using optimi's accumulation in a framework like Composer
        if random.random() < 0.5:
            optimi_optimizer.step()
            optimi_optimizer.zero_grad()

    assert_most_approx_close(m1.fc1.weight, m2.fc1.weight, rtol=tolerance.rtol, atol=tolerance.atol, max_error_rate=max_error_rate)
    assert_most_approx_close(m1.fc2.weight, m2.fc2.weight, rtol=tolerance.rtol, atol=tolerance.atol, max_error_rate=max_error_rate)

    remove_gradient_release(m2)
