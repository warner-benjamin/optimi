from itertools import product

import pytest
import torch

import optimi
from packaging.version import parse

from tests.optimizer_test import (buffer, run_optimizer, gradient_release, cpu_dim1, cpu_dim2, cpu_gtype,
                                  cpu_ftype, cuda_dim1, cuda_dim2, cuda_gtype, cuda_ftype, gr_dim1,
                                  gr_dim2, gr_dtype, gr_ftype, optimizer_accumulation, cuda_device)

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



cpu_values = list(product(cpu_dim1, cpu_dim2, cpu_gtype, optimizer_names, cpu_ftype))
cpu_names = ["dim1_{}_dim2_{}_gtype_{}_optim_{}{}".format(*vals) for vals in cpu_values]

@pytest.mark.cpu
@pytest.mark.radam
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name, ftype", cpu_values, ids=cpu_names)
def test_optimizer_cpu(dim1:int, dim2:int, gtype:torch.dtype, optim_name:str, ftype:str):
    run_optimizer(optimizers, dim1, dim2, gtype, optim_name, ftype, torch.device('cpu'), buffer)



cuda_values = list(product(cuda_dim1, cuda_dim2, cuda_gtype, optimizer_names, cuda_ftype))
cuda_names = ["dim1_{}_dim2_{}_gtype_{}_optim_{}{}".format(*vals) for vals in cuda_values]

@pytest.mark.cuda
@pytest.mark.radam
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_optimizer_cuda(dim1:int, dim2:int, gtype:torch.dtype, optim_name:str, ftype:str, cuda_device:str):
    run_optimizer(optimizers, dim1, dim2, gtype, optim_name, ftype, torch.device(cuda_device), buffer, iterations=80)



cuda_values = list(product(gr_dim1, gr_dim2, gr_dtype, optimizer_names, gr_ftype))
cuda_names = ["dim1_{}_dim2_{}_gtype_{}_optim_{}{}".format(*vals) for vals in cuda_values]

@pytest.mark.cuda
@pytest.mark.radam
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_gradient_release(dim1:int, dim2:int, gtype:torch.dtype, optim_name:str, ftype:str, cuda_device:str):
    gradient_release(optimizers, dim1, dim2, gtype, optim_name, ftype, torch.device(cuda_device),
                     framework_opt_step=torch.rand(1).item() > 0.5)


@pytest.mark.cuda
@pytest.mark.radam
@pytest.mark.parametrize("dim1, dim2, gtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_optimizer_accumulation(dim1:int, dim2:int, gtype:torch.dtype, optim_name:str, ftype:str, cuda_device:str):
    if optim_name in ["radam_l2"]:
        pytest.skip("Skip tests for RAdam with L2 weight decay.")
    optimizer_accumulation(optimizers, dim1, dim2, gtype, optim_name, ftype, torch.device(cuda_device),
                           framework_opt_step=torch.rand(1).item() > 0.5)
