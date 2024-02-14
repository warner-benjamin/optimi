# Optimizer testing modified from bitsandbytes: https://github.com/TimDettmers/bitsandbytes/blob/main/tests/test_optim.py
# bitsandbytes - MIT License - Copyright (c) Facebook, Inc. and its affiliates.

import inspect
import io

import torch

from optimi import prepare_for_gradient_release, remove_gradient_release


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, device, dtype):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act = torch.nn.Mish()
        self.fc2 = torch.nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def assert_most_approx_close(a, b, rtol=1e-3, atol=1e-3, max_error_count=0, max_error_rate=None):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()
    if max_error_rate is not None:
        if error_count > (a.shape[0] * a.shape[1]) * max_error_rate and error_count > max_error_count:
            print(f"Too many values not close: assert {error_count} < {(a.shape[0] * a.shape[1]) * max_error_rate}")
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    elif error_count > max_error_count:
        print(f"Too many values not close: assert {error_count} < {max_error_count}")
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def load_optimizer(params, optimizers, optim_name, key, ftype) -> torch.optim.Optimizer:
    def update_kwargs(key, argspec, value=True):
        if key in argspec.kwonlyargs or key in argspec.args:
            kwargs.update({key: value})

    if optim_name in optimizers:
        optimizer = optimizers[optim_name][key]['optim']
        kwargs =    optimizers[optim_name][key]['kwargs']
    else:
        raise ValueError(f"{optim_name} optimizer not defined")

    argspec = inspect.getfullargspec(optimizer)
    if ftype != '':
        update_kwargs(ftype, argspec)
    else:
        update_kwargs('fused', argspec, False)
        update_kwargs('foreach', argspec, False)

    return optimizer(params, **kwargs)


def run_optimizer(optimizers:dict, dim1:int, dim2:int, gtype:torch.dtype, optim_name:str,
                  ftype:str, device:torch.device, buffer:io.BytesIO, iterations:int=20,
                  any_precision:bool=False):
    if dim1 == 1 and dim2 == 1:
        return
    ftype = ftype.replace('_', '')

    # since Lion can have pretty noisy updates where things lie at the boundary
    # allow up to 10 errors for Lion on GPU size tensors and 2 on CPU size tensors
    max_error_count = 2 if device == torch.device('cpu') else 10
    # Adan bfloat16 updates are noisier than other optimizers,
    # allow more errors for higher dimension testing
    if dim2 >= 4096:
        max_error_count *= 3

    p1 = torch.randn(dim1, dim2, device=device, dtype=gtype) * 0.1
    p2 = p1.clone()
    if not any_precision:
        p1 = p1.float()

    torch_optimizer = load_optimizer([p1], optimizers, optim_name, 0, ftype)
    optimi_optimizer = load_optimizer([p2], optimizers, optim_name, 1, ftype)

    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    elif gtype == torch.bfloat16 and not any_precision:
        atol, rtol = 1e-3, 1e-2
    elif gtype == torch.float16 or (gtype == torch.bfloat16 and any_precision):
        atol, rtol = 1e-4, 1e-3

    for i in range(iterations):
        g = torch.randn(dim1, dim2, device=device, dtype=gtype) * 0.01
        if any_precision:
            p1.grad = g.clone()
        else:
            p1.grad = g.clone().float()
        p2.grad = g.clone()

        optimi_optimizer.step()
        torch_optimizer.step()

        if any_precision:
            # allow more errors for any_precision since the tolerances are more
            # precise then the regular bfloat16 precision tests
            assert_most_approx_close(p1.float(), p2.float(), atol, rtol,
                                     max_error_count=max_error_count, max_error_rate=0.001)
        else:
            assert_most_approx_close(p1, p2.float(), atol, rtol, max_error_count=max_error_count)

        if i % (iterations // 10) == 0 and i > 0:
            torch.save(optimi_optimizer.state_dict(), buffer)
            buffer.seek(0)
            del optimi_optimizer
            optimi_optimizer = None
            optimi_optimizer = load_optimizer([p2], optimizers, optim_name, 1, ftype)
            optimi_optimizer.load_state_dict(torch.load(buffer))
            buffer.seek(0)
            buffer.truncate(0)
            if any_precision:
                assert_most_approx_close(p1.float(), p2.float(), atol, rtol,
                                         max_error_count=max_error_count, max_error_rate=0.001)
            else:
                assert_most_approx_close(p1, p2.float(), atol, rtol, max_error_count=max_error_count)


def gradient_release(optimizers:dict, dim1:int, dim2:int, dtype:torch.dtype, optim_name:str,
                    ftype:str, device:torch.device, iterations:int=20, framework_opt_step:bool=False):
    def optimizer_hook(parameter) -> None:
        torch_optimizers[parameter].step()
        torch_optimizers[parameter].zero_grad()

    # Since Lion & Adan can have noisy updates, allow up to 10 errors
    max_error_count = 10

    if dtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    elif dtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-2
    elif dtype == torch.float16:
        atol, rtol = 1e-4, 1e-3

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


    # Training loop
    for i in range(iterations):
        input1 = torch.randn(1, dim1, device=device, dtype=dtype)
        input2 = input1.clone()
        input3 = input1.clone()
        target1 = torch.randn(1, 1, device=device, dtype=dtype)
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

        assert_most_approx_close(m1.fc1.weight, m2.fc1.weight, rtol=rtol, atol=atol, max_error_count=max_error_count)
        assert_most_approx_close(m1.fc2.weight, m2.fc2.weight, rtol=rtol, atol=atol, max_error_count=max_error_count)
        assert_most_approx_close(m1.fc1.weight, m3.fc1.weight, rtol=rtol, atol=atol, max_error_count=max_error_count)
        assert_most_approx_close(m1.fc2.weight, m3.fc2.weight, rtol=rtol, atol=atol, max_error_count=max_error_count)

    for h in pytorch_hooks:
        h.remove()
    remove_gradient_release(m3)


buffer = io.BytesIO()


cpu_dim1 = [256]
cpu_dim2 = [32, 1]
cpu_gtype = [torch.float32]
cpu_ftype = ['', '_foreach']


cuda_dim1 = [1024]
cuda_dim2 = [64, 1024, 4096, 1]
cuda_gtype = [torch.float32, torch.bfloat16]
cuda_ftype = ['', '_foreach']

gr_dim1 = [128]
gr_dim2 = [256, 1024]
gr_dtype = [torch.float32]
gr_ftype = ['']