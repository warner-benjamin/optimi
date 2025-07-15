from itertools import product

import pytest
import torch

import optimi
from packaging.version import parse

from tests.optimizer_test import (buffer, run_optimizer, gradient_release, cpu_dim1, cpu_dim2, cpu_dtype,
                                  cpu_ftype, gpu_dim1, gpu_dim2, gpu_dtype, gpu_ftype, gr_dim1,
                                  gr_dim2, gr_dtype, gr_ftype, optimizer_accumulation, gpu_device)

# PyTorch's RAdam adds epsilon before debiasing V while Optimi debases before.
# RAdam tests with a smaller epsilon then other optimizers to prevent numerical divergances.

optimizers = {}

optimizers["radam"] = ({'optim':torch.optim.RAdam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0)},
                       {'optim':optimi.RAdam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0)})

optimizers["radam_l2"] = ({'optim':torch.optim.RAdam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2)},
                          {'optim':optimi.RAdam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2, decouple_wd=False)})

if parse(torch.__version__) >= parse("2.2"):
    optimizers["radamw"] = ({'optim':torch.optim.RAdam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2, decoupled_weight_decay=True)},
                            {'optim':optimi.RAdam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2, decouple_wd=True)})

    optimizers["radam_dlr"] = ({'optim':torch.optim.RAdam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-2, decoupled_weight_decay=True)},
                               {'optim':optimi.RAdam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-5, decouple_lr=True)})

optimizer_names = [key for key in optimizers.keys()]



cpu_values = list(product(cpu_dim1, cpu_dim2, cpu_dtype, optimizer_names, cpu_ftype))
cpu_names = ["dim1_{}_dim2_{}_dtype_{}_optim_{}{}".format(*vals) for vals in cpu_values]

@pytest.mark.cpu
@pytest.mark.radam
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cpu_values, ids=cpu_names)
def test_optimizer_cpu(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str):
    run_optimizer(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device('cpu'), buffer)



cuda_values = list(product(gpu_dim1, gpu_dim2, gpu_dtype, optimizer_names, gpu_ftype))
cuda_names = ["dim1_{}_dim2_{}_dtype_{}_optim_{}{}".format(*vals) for vals in cuda_values]

@pytest.mark.gpu
@pytest.mark.radam
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_optimizer_gpu(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str, gpu_device:str):
    run_optimizer(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device(gpu_device), buffer, max_error_rate_override={torch.float32: 0.001})



cuda_values = list(product(gr_dim1, gr_dim2, gr_dtype, optimizer_names, gr_ftype))
cuda_names = ["dim1_{}_dim2_{}_dtype_{}_optim_{}{}".format(*vals) for vals in cuda_values]

@pytest.mark.gpu
@pytest.mark.radam
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_gradient_release(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str, gpu_device:str):
    gradient_release(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device(gpu_device),
                     framework_opt_step=torch.rand(1).item() > 0.5, max_error_rate_override={torch.float32: 0.001})


@pytest.mark.gpu
@pytest.mark.radam
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_optimizer_accumulation(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str, gpu_device:str):
    if optim_name in ["radam_l2"]:
        pytest.skip("Skip tests for RAdam with L2 weight decay.")
    optimizer_accumulation(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device(gpu_device),
                           framework_opt_step=torch.rand(1).item() > 0.5, max_error_rate_override={torch.float32: 0.001})
