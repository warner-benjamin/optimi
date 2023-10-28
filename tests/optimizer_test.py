# Optimizer testing modified from bitsandbytes: https://github.com/TimDettmers/bitsandbytes/blob/main/tests/test_optim.py
# bitsandbytes - MIT License - Copyright (c) Facebook, Inc. and its affiliates.

import inspect
import io

import torch


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
    max_error_count = 10 if device != torch.device('cpu') else 1

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

        # since Lion can have pretty noisy updates where things lie at the boundary
        # allow up to 10 errors for Lion on GPU size tensors and 1 on CPU size tensors
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



buffer = io.BytesIO()


cpu_dim1 = [256]
cpu_dim2 = [32, 1]
cpu_gtype = [torch.float32]
cpu_ftype = ['', '_foreach']


cuda_dim1 = [1024]
cuda_dim2 = [64, 1024, 4097, 1]
cuda_gtype = [torch.float32, torch.bfloat16]
cuda_ftype = ['', '_foreach']