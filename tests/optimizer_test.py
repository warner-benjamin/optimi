# Optimizer testing modified from bitsandbytes: https://github.com/TimDettmers/bitsandbytes/blob/main/tests/test_optim.py
# bitsandbytes - MIT License - Copyright (c) Facebook, Inc. and its affiliates.

import inspect
import io
from itertools import product

import pytest
import torch

import optimi
from tests import reference

k = 20


optimizers, any_optimizers = {}, {}

optimizers["adam"] = ({'optim':torch.optim.Adam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6)},
                      {'optim':optimi.Adam, 'kwargs':dict(lr=1e-3, weight_decay=0)})

optimizers["adam_l2"] = ({'optim':torch.optim.Adam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2)},
                         {'optim':optimi.Adam, 'kwargs':dict(lr=1e-3, weight_decay=1e-2, decouple_wd=False)})

optimizers["adam_dw"] = ({'optim':torch.optim.AdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2)},
                         {'optim':optimi.Adam, 'kwargs':dict(lr=1e-3, weight_decay=1e-2, decouple_wd=True)})

optimizers["adamw"] = ({'optim':torch.optim.AdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2)},
                       {'optim':optimi.AdamW, 'kwargs':dict(lr=1e-3, weight_decay=1e-2)})

optimizers["adamw_dlr"] = ({'optim':reference.DecoupledAdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-5)},
                           {'optim':optimi.AdamW, 'kwargs':dict(lr=1e-3, weight_decay=1e-5, decouple_lr=True)})

optimizers["sgd"] = ({'optim':torch.optim.SGD, 'kwargs':dict(lr=1e-3)},
                     {'optim':optimi.SGD, 'kwargs':dict(lr=1e-3)})

optimizers["sgd_mom"] = ({'optim':torch.optim.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9)},
                         {'optim':optimi.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=False)})

optimizers["sgd_damp"] = ({'optim':torch.optim.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=0.9)},
                          {'optim':optimi.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=True, torch_init=True)})

optimizers["sgd_l2"] = ({'optim':torch.optim.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, weight_decay=1e-2)},
                        {'optim':optimi.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=False, weight_decay=1e-2, decouple_wd=False)})

optimizers["sgdw_dlr"] = ({'optim':reference.DecoupledSGDW, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=0.9, weight_decay=1e-5)},
                          {'optim':optimi.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=True, torch_init=True, decouple_lr=True, weight_decay=1e-5)})

any_optimizers["any_adamw"] = ({'optim':reference.AnyPrecisionAdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2)},
                               {'optim':optimi.Adam, 'kwargs':dict(lr=1e-3, weight_decay=1e-2, decouple_wd=True, kahan_sum=True)})


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


def load_optimizer(params, optim_name, key, ftype) -> torch.optim.Optimizer:
    def update_kwargs(key, argspec, value=True):
        if key in argspec.kwonlyargs or key in argspec.args:
            kwargs.update({key: value})

    if optim_name in optimizers:
        optimizer = optimizers[optim_name][key]['optim']
        kwargs =    optimizers[optim_name][key]['kwargs']
    elif optim_name in any_optimizers:
        optimizer = any_optimizers[optim_name][key]['optim']
        kwargs =    any_optimizers[optim_name][key]['kwargs']
    else:
        raise ValueError(f"{optim_name} optimizer not defined")

    argspec = inspect.getfullargspec(optimizer)
    if ftype != '':
        update_kwargs(ftype, argspec)
    else:
        update_kwargs('fused', argspec, False)
        update_kwargs('foreach', argspec, False)

    return optimizer(params, **kwargs)


def run_optimizer(dim1:int, dim2:int, gtype:torch.dtype, optim_name:str,
                  ftype:str, device:torch.device, buffer:io.BytesIO,
                  any_precision:bool=False):
    if dim1 == 1 and dim2 == 1:
        return
    max_error_count = 10 if device != torch.device('cpu') else 1

    p1 = torch.randn(dim1, dim2, device=device, dtype=gtype) * 0.1
    p2 = p1.clone()
    if not any_precision:
        p1 = p1.float()

    torch_optimizer = load_optimizer([p1], optim_name, 0, ftype)
    optimi_optimizer = load_optimizer([p2], optim_name, 1, ftype)

    if gtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    elif gtype == torch.bfloat16 and not any_precision:
        atol, rtol = 1e-3, 1e-2
    elif gtype == torch.float16 or (gtype == torch.bfloat16 and any_precision):
        atol, rtol = 1e-4, 1e-3

    for i in range(k):
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

        if i % (k // 10) == 0 and i > 0:
            torch.save(optimi_optimizer.state_dict(), buffer)
            buffer.seek(0)
            del optimi_optimizer
            optimi_optimizer = None
            optimi_optimizer = load_optimizer([p2], optim_name, 1, ftype)
            optimi_optimizer.load_state_dict(torch.load(buffer))
            buffer.seek(0)
            buffer.truncate(0)
            if any_precision:
                assert_most_approx_close(p1.float(), p2.float(), atol, rtol,
                                         max_error_count=max_error_count, max_error_rate=0.001)
            else:
                assert_most_approx_close(p1, p2.float(), atol, rtol, max_error_count=max_error_count)


buffer = io.BytesIO()

optimizer_names = [key for key in optimizers.keys()]
cpu_dim1 = [256]
cpu_dim2 = [32, 1]
cpu_gtype = [torch.float32]
cpu_ftype = ['', '_foreach']
cpu_values = list(product(cpu_dim1, cpu_dim2, cpu_gtype, optimizer_names, cpu_ftype))
cpu_names = ["dim1_{}_dim2_{}_gtype_{}_optim_{}{}".format(*vals) for vals in cpu_values]

@pytest.mark.cpu
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name, ftype", cpu_values, ids=cpu_names)
def test_optimizer_cpu(dim1:int, dim2:int, gtype:torch.dtype, optim_name:str, ftype:str):
    ftype = ftype.replace('_', '')
    run_optimizer(dim1, dim2, gtype, optim_name, ftype, torch.device('cpu'), buffer)


optimizer_names = [key for key in any_optimizers.keys()]
cpu_gtype = [torch.bfloat16]
cpu_values = list(product(cpu_dim1, cpu_dim2, cpu_gtype, optimizer_names, cpu_ftype))
cpu_names = ["dim1_{}_dim2_{}_gtype_{}_optim_{}{}".format(*vals) for vals in cpu_values]

@pytest.mark.cpu
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name, ftype", cpu_values, ids=cpu_names)
def test_anyoptimizer_cpu(dim1:int, dim2:int, gtype:torch.dtype, optim_name:str, ftype:str):
    ftype = ftype.replace('_', '')
    run_optimizer(dim1, dim2, gtype, optim_name, ftype, torch.device('cpu'), buffer, True)


optimizer_names = [key for key in optimizers.keys()]
cuda_dim1 = [1024]
cuda_dim2 = [64, 1024, 4097, 1]
cuda_gtype = [torch.float32, torch.bfloat16]
cuda_ftype = ['', '_foreach']
cuda_values = list(product(cuda_dim1, cuda_dim2, cuda_gtype, optimizer_names, cuda_ftype))
cuda_names = ["dim1_{}_dim2_{}_gtype_{}_optim_{}{}".format(*vals) for vals in cuda_values]

@pytest.mark.cuda
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_optimizer_cuda(dim1:int, dim2:int, gtype:torch.dtype, optim_name:str, ftype:str):
    ftype = ftype.replace('_', '')
    run_optimizer(dim1, dim2, gtype, optim_name, ftype, torch.device('cuda'), buffer)


optimizer_names = [key for key in any_optimizers.keys()]
cuda_gtype = [torch.bfloat16]
cuda_values = list(product(cuda_dim1, cuda_dim2, cuda_gtype, optimizer_names, cuda_ftype))
cuda_names = ["dim1_{}_dim2_{}_gtype_{}_optim_{}{}".format(*vals) for vals in cuda_values]

@pytest.mark.cuda
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_anyoptimizer_cuda(dim1:int, dim2:int, gtype:torch.dtype, optim_name:str, ftype:str):
    ftype = ftype.replace('_', '')
    run_optimizer(dim1, dim2, gtype, optim_name, ftype, torch.device('cuda'), buffer, True)