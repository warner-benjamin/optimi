from itertools import product

import pytest
import torch

import optimi
from tests import reference

from tests.optimizer_test import (buffer, run_optimizer, gradient_release, cpu_dim1, cpu_dim2, cpu_dtype,
                                  cpu_ftype, gpu_dim1, gpu_dim2, gpu_dtype, gpu_ftype, gr_dim1,
                                  gr_dim2, gr_dtype, gr_ftype, optimizer_accumulation, gpu_device)


optimizers = {}

optimizers["adam"] = ({'optim':torch.optim.Adam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0)},
                      {'optim':optimi.Adam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0)})

optimizers["adam_l2"] = ({'optim':torch.optim.Adam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2)},
                         {'optim':optimi.Adam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2, decouple_wd=False)})

optimizers["adam_dw"] = ({'optim':torch.optim.AdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2)},
                         {'optim':optimi.Adam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2, decouple_wd=True)})

optimizers["adamw"] = ({'optim':torch.optim.AdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2)},
                       {'optim':optimi.AdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2)})

optimizers["adamw_dlr"] = ({'optim':reference.DecoupledAdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-5)},
                           {'optim':optimi.AdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-5, decouple_lr=True)})

optimizer_names = [key for key in optimizers.keys()]



cpu_values = list(product(cpu_dim1, cpu_dim2, cpu_dtype, optimizer_names, cpu_ftype))
cpu_names = ["dim1_{}_dim2_{}_dtype_{}_optim_{}{}".format(*vals) for vals in cpu_values]

@pytest.mark.cpu
@pytest.mark.adam
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cpu_values, ids=cpu_names)
def test_optimizer_cpu(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str):
    run_optimizer(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device('cpu'), buffer)



cuda_values = list(product(gpu_dim1, gpu_dim2, gpu_dtype, optimizer_names, gpu_ftype))
cuda_names = ["dim1_{}_dim2_{}_dtype_{}_optim_{}{}".format(*vals) for vals in cuda_values]

@pytest.mark.gpu
@pytest.mark.adam
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_optimizer_gpu(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str, gpu_device:str):
    run_optimizer(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device(gpu_device), buffer)



cuda_values = list(product(gr_dim1, gr_dim2, gr_dtype, optimizer_names, gr_ftype))
cuda_names = ["dim1_{}_dim2_{}_dtype_{}_optim_{}{}".format(*vals) for vals in cuda_values]

@pytest.mark.gpu
@pytest.mark.adam
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_gradient_release(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str, gpu_device:str):
    gradient_release(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device(gpu_device),
                     framework_opt_step=torch.rand(1).item() > 0.5)


@pytest.mark.gpu
@pytest.mark.adam
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_optimizer_accumulation(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str, gpu_device:str):
    if optim_name in ["adam_l2"]:
        pytest.skip("Skip tests for Adam with L2 weight decay.")
    optimizer_accumulation(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device(gpu_device),
                           framework_opt_step=torch.rand(1).item() > 0.5)