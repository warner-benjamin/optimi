# Optimizer testing modified from bitsandbytes: https://github.com/TimDettmers/bitsandbytes/blob/main/tests/test_optim.py
# bitsandbytes - MIT License - Copyright (c) Facebook, Inc. and its affiliates.

import inspect
import io
from typing import Optional

import pytest
import torch
from torch import Tensor
from optimi.utils import MIN_TORCH_2_6
from optimi import prepare_for_gradient_release, remove_gradient_release


@pytest.fixture()
def gpu_device(worker_id, request):
    """Map xdist workers to available GPU devices in a round-robin fashion,
    supporting CUDA (NVIDIA/ROCm) and XPU (Intel) backends.
    Use a single specified GPU if --gpu-id is provided"""

    # Check if specific GPU was requested
    specific_gpu = request.config.getoption("--gpu-id")

    # Determine available GPU backend and device count
    if torch.cuda.is_available():
        backend = "cuda"
        device_count = torch.cuda.device_count()
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        backend = "xpu"
        device_count = torch.xpu.device_count()
    else:
        # Fallback to cuda for compatibility
        backend = "cuda"
        device_count = 0

    if specific_gpu is not None:
        return f"{backend}:{specific_gpu}"

    if worker_id == "master":
        return backend

    # If no devices available, return default backend
    if device_count == 0:
        return backend

    # Extract worker number from worker_id (e.g., 'gw6' -> 6)
    worker_num = int(worker_id.replace('gw', ''))

    # Map worker to GPU index using modulo to round-robin
    gpu_idx = (worker_num - 1) % device_count
    return f"{backend}:{gpu_idx}"


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, device, dtype):
        super().__init__()
        self.norm = torch.nn.LayerNorm(input_size, device=device, dtype=dtype)
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act = torch.nn.Mish()
        self.fc2 = torch.nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def assert_most_approx_close(a: Tensor, b: Tensor, rtol: float = 1e-3, atol: float = 1e-3, max_error_count: int = 0, max_error_rate: float | None = None, name: str = ''):
    idx = torch.isclose(a.float(), b.float(), rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()
    if max_error_rate is not None:
        if error_count > (a.numel()) * max_error_rate and error_count > max_error_count:
            print(f"{name}Too many values not close: assert {error_count} < {(a.numel()) * max_error_rate}")
            torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol)
    elif error_count > max_error_count:
        print(f"{name}Too many values not close: assert {error_count} < {max_error_count}")
        torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol)


def load_optimizer(params, optimizers, optim_name, key, ftype, skip=False) -> torch.optim.Optimizer:
    def update_kwargs(key, argspec, value=True):
        if key in argspec.kwonlyargs or key in argspec.args:
            kwargs.update({key: value})
        elif value and skip:
            pytest.skip(f"Skipping {key} for {optim_name}")

    if optim_name in optimizers:
        optimizer = optimizers[optim_name][key]['optim']
        kwargs =    optimizers[optim_name][key]['kwargs']
    else:
        raise ValueError(f"{optim_name} optimizer not defined")

    argspec = inspect.getfullargspec(optimizer)
    update_kwargs('fused', argspec, False)
    update_kwargs('foreach', argspec, False)
    update_kwargs('triton', argspec, False)
    if ftype != '':
        update_kwargs(ftype, argspec, True)

    return optimizer(params, **kwargs)


def run_optimizer(optimizers:dict, dim1:int, dim2:int, dtype:torch.dtype, optim_name:str,
                  ftype:str, device:torch.device, buffer:io.BytesIO, iterations:Optional[int]=None,
                  any_precision:bool=False, atol_override:Optional[dict[torch.dtype, float]]=None,
                  rtol_override:Optional[dict[torch.dtype, float]]=None,
                  max_error_rate_override:Optional[dict[torch.dtype, float]]=None):
    if dim1 == 1 and dim2 == 1:
        pytest.skip("Skipping 1x1 optimizer test")

    ftype = ftype.replace('_', '')

    if atol_override is None:
        atol_override = {}
    if rtol_override is None:
        rtol_override = {}
    if max_error_rate_override is None:
        max_error_rate_override = {}

    if iterations is None:
        if device == torch.device('cpu'):
            iterations = 20
        else:
            iterations = 40

    # # since Lion can have pretty noisy updates where things lie at the boundary
    # # allow up to 10 errors for Lion on GPU size tensors and 2 on CPU size tensors
    # max_error_count = 2 if device == torch.device('cpu') else 10
    # # Adan bfloat16 updates are noisier than other optimizers,
    # # allow more errors for higher dimension testing
    # if dim2 >= 2048:
    #     max_error_count *= 3
    max_error_count = 0

    if dtype == torch.float32:
        atol = atol_override.get(torch.float32, 1e-6)
        rtol = rtol_override.get(torch.float32, 1e-5)
        max_error_rate = max_error_rate_override.get(torch.float32, 0.0001)
    elif dtype == torch.bfloat16:
        atol = atol_override.get(torch.bfloat16, 1e-3)
        rtol = rtol_override.get(torch.bfloat16, 1e-2)
        max_error_rate = max_error_rate_override.get(torch.bfloat16, 0.01)
    elif dtype == torch.float16:
        atol = atol_override.get(torch.float16, 1e-4)
        rtol = rtol_override.get(torch.float16, 1e-3)
        max_error_rate = max_error_rate_override.get(torch.float16, 0.01)

    # Create MLP models instead of simple parameters
    m1 = MLP(dim1, dim2, device=device, dtype=dtype)
    m2 = MLP(dim1, dim2, device=device, dtype=dtype)
    m2.load_state_dict(m1.state_dict())

    # Convert model parameters to float for non-any_precision testing
    if not any_precision and dtype != torch.float32:
        for p in m1.parameters():
            p.data = p.data.float()

    torch_optimizer = load_optimizer(m1.parameters(), optimizers, optim_name, 0, ftype)
    optimi_optimizer = load_optimizer(m2.parameters(), optimizers, optim_name, 1, ftype, skip=True)

    bs = 1 if device.type == "cpu" else 32

    for i in range(iterations):
        # Training loop with input/target generation
        input1 = torch.randn(bs, dim1, device=device, dtype=dtype)
        input2 = input1.detach().clone()
        target1 = torch.randn(bs, 1, device=device, dtype=dtype)
        target2 = target1.detach().clone()

        # Convert model parameters to float for non-any_precision testing
        if not any_precision and dtype != torch.float32:
            input1 = input1.float()
            target1 = target1.float()

        # Forward pass
        output1 = m1(input1)
        output2 = m2(input2)

        # Loss calculation
        loss1 = torch.nn.functional.mse_loss(output1, target1)
        loss2 = torch.nn.functional.mse_loss(output2, target2)

        # Backward pass
        loss1.backward()
        loss2.backward()

        # Optimizer step
        optimi_optimizer.step()
        torch_optimizer.step()

        # Zero gradients
        optimi_optimizer.zero_grad()
        torch_optimizer.zero_grad()

        # Compare model weights
        assert_most_approx_close(m1.fc1.weight, m2.fc1.weight, atol=atol, rtol=rtol,
                                max_error_count=max_error_count, max_error_rate=max_error_rate,
                                name='fc1: ')
        assert_most_approx_close(m1.fc2.weight, m2.fc2.weight, atol=atol, rtol=rtol,
                                max_error_count=max_error_count, max_error_rate=max_error_rate,
                                name='fc2: ')

        # # Test state_dict saving and loading periodically
        if i % (iterations // 10) == 0 and i > 0:
            # Save optimizer state
            torch.save(optimi_optimizer.state_dict(), buffer)
            buffer.seek(0)
            # Load checkpoint
            ckpt = torch.load(buffer, weights_only=True)
            # Recreate optimizer and load its state
            optimi_optimizer = load_optimizer(m2.parameters(), optimizers, optim_name, 1, ftype)
            optimi_optimizer.load_state_dict(ckpt)
            # Clear buffer
            buffer.seek(0)
            buffer.truncate(0)

            # Verify models are still aligned after state_dict loading
            assert_most_approx_close(m1.fc1.weight, m2.fc1.weight, atol=atol, rtol=rtol,
                                     max_error_count=max_error_count, max_error_rate=max_error_rate,
                                     name='fc1 after load: ')
            assert_most_approx_close(m1.fc2.weight, m2.fc2.weight, atol=atol, rtol=rtol,
                                     max_error_count=max_error_count, max_error_rate=max_error_rate,
                                     name='fc2 after load: ')


def gradient_release(optimizers:dict, dim1:int, dim2:int, dtype:torch.dtype, optim_name:str,
                     ftype:str, device:torch.device, iterations:int=40, framework_opt_step:bool=False,
                     atol_override:Optional[dict[torch.dtype, float]]=None,
                     rtol_override:Optional[dict[torch.dtype, float]]=None,
                     max_error_rate_override:Optional[dict[torch.dtype, float]]=None):
    def optimizer_hook(parameter) -> None:
        torch_optimizers[parameter].step()
        torch_optimizers[parameter].zero_grad()

    # Since Lion & Adan can have noisy updates, allow up to 12 errors
    max_error_count = 12

    if atol_override is None:
        atol_override = {}
    if rtol_override is None:
        rtol_override = {}
    if max_error_rate_override is None:
        max_error_rate_override = {}

    if dtype == torch.float32:
        atol = atol_override.get(torch.float32, 2e-6)
        rtol = rtol_override.get(torch.float32, 1e-5)
    elif dtype == torch.bfloat16:
        atol = atol_override.get(torch.bfloat16, 2e-3)
        rtol = rtol_override.get(torch.bfloat16, 1e-2)
    elif dtype == torch.float16:
        atol = atol_override.get(torch.float16, 2e-4)
        rtol = rtol_override.get(torch.float16, 1e-3)

    m1 = MLP(dim1, dim2, device=device, dtype=dtype)
    m2 = MLP(dim1, dim2, device=device, dtype=dtype)
    m3 = MLP(dim1, dim2, device=device, dtype=dtype)
    m2.load_state_dict(m1.state_dict())
    m3.load_state_dict(m1.state_dict())

    regular_optimizer = load_optimizer(m1.parameters(), optimizers, optim_name, 0, ftype)


    # PyTorch Method: taken from https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
    torch_optimizers = {p: load_optimizer([p], optimizers, optim_name, 0, ftype) for p in m2.parameters()}

    pytorch_hooks = []
    for p in m2.parameters():
        pytorch_hooks.append(p.register_post_accumulate_grad_hook(optimizer_hook))


    # Optimim Method
    # add the gradient release flag to the optimizer kwargs
    optimizers[optim_name][1]['kwargs']['gradient_release'] = True
    optimi_optimizer = load_optimizer(m3.parameters(), optimizers, optim_name, 1, ftype)

    prepare_for_gradient_release(m3, optimi_optimizer)
    bs = 1 if device.type == "cpu" else 32


    # Training loop
    for i in range(iterations):
        input1 = torch.randn(bs, dim1, device=device, dtype=dtype)
        input2 = input1.clone()
        input3 = input1.clone()
        target1 = torch.randn(bs, 1, device=device, dtype=dtype)
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

        # simulates using an optimi gradient release optimizer in a framework
        # where the optimizer step and zero_grad cannot be disabled.
        if framework_opt_step:
            optimi_optimizer.step()
            optimi_optimizer.zero_grad()

        assert_most_approx_close(m1.fc1.weight, m2.fc1.weight, rtol=rtol, atol=atol,
                                 max_error_count=max_error_count, name='PyTorch-PyTorch: ')
        assert_most_approx_close(m1.fc2.weight, m2.fc2.weight, rtol=rtol, atol=atol,
                                 max_error_count=max_error_count, name='PyTorch-PyTorch: ')
        assert_most_approx_close(m1.fc1.weight, m3.fc1.weight, rtol=rtol, atol=atol,
                                 max_error_count=max_error_count, name='PyTorch-Optimi: ')
        assert_most_approx_close(m1.fc2.weight, m3.fc2.weight, rtol=rtol, atol=atol,
                                 max_error_count=max_error_count, name='PyTorch-Optimi: ')

    for h in pytorch_hooks:
        h.remove()
    remove_gradient_release(m3)


def optimizer_accumulation(optimizers:dict, dim1:int, dim2:int, dtype:torch.dtype, optim_name:str,
                           ftype:str, device:torch.device, iterations:int=40, framework_opt_step:bool=False,
                           atol_override:Optional[dict[torch.dtype, float]]=None,
                           rtol_override:Optional[dict[torch.dtype, float]]=None,
                           max_error_rate_override:Optional[dict[torch.dtype, float]]=None):
    # Since optimizer accumulation approximates gradient accumulation, the tolerances
    # compared to normal optimizers are high despite the low number of iterations
    max_error_rate = 0.035
    atol, rtol = 1e-2, 1e-2

    m1 = MLP(dim1, dim2, device=device, dtype=dtype)
    m2 = MLP(dim1, dim2, device=device, dtype=dtype)
    m2.load_state_dict(m1.state_dict())

    regular_optimizer = load_optimizer(m1.parameters(), optimizers, optim_name, 0, ftype)


    # Optimim Method
    # add the gradient release flag to the optimizer kwargs
    optimizers[optim_name][1]['kwargs']['gradient_release'] = True
    optimi_optimizer = load_optimizer(m2.parameters(), optimizers, optim_name, 1, ftype)

    prepare_for_gradient_release(m2, optimi_optimizer)

    gradient_accumulation_steps = 4
    bs = 1 if device.type == "cpu" else 32

    # Training loop
    for i in range(iterations):
        input1 = torch.randn(bs, dim1, device=device, dtype=dtype)
        input2 = input1.clone()
        target1 = torch.randn(bs, 1, device=device, dtype=dtype)
        target2 = target1.clone()

        optimi_optimizer.optimizer_accumulation = (i+1) % gradient_accumulation_steps != 0

        output1 = m1(input1)
        output2 = m2(input2)

        loss1 = torch.nn.functional.mse_loss(output1, target1)
        loss2 = torch.nn.functional.mse_loss(output2, target2)

        loss1.backward()
        loss2.backward()

        if not optimi_optimizer.optimizer_accumulation:
            regular_optimizer.step()
            regular_optimizer.zero_grad()

        # simulates using an optimi gradient release optimizer in a framework
        # where the optimizer step and zero_grad cannot be disabled.
        if framework_opt_step:
            optimi_optimizer.step()
            optimi_optimizer.zero_grad()

    # unlike other tests, compare that the weights are in the same approximate range at the end of training
    assert_most_approx_close(m1.fc1.weight, m2.fc1.weight, rtol=rtol, atol=atol, max_error_rate=max_error_rate)
    assert_most_approx_close(m1.fc2.weight, m2.fc2.weight, rtol=rtol, atol=atol, max_error_rate=max_error_rate)

    remove_gradient_release(m2)


buffer = io.BytesIO()


cpu_dim1 = [64]
cpu_dim2 = [64, 128]
cpu_dtype = [torch.float32]
cpu_ftype = ['', '_foreach']


gpu_dim1 = [256]
gpu_dim2 = [256, 512, 1024, 2048]
gpu_dtype = [torch.float32, torch.bfloat16]
gpu_ftype = ['', '_foreach'] + (['_triton'] if MIN_TORCH_2_6 else [])

gr_dim1 = [128]
gr_dim2 = [256, 1024]
gr_dtype = [torch.float32]
gr_ftype = [''] + (['_triton'] if MIN_TORCH_2_6 else [])