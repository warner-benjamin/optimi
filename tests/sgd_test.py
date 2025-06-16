from itertools import product

import pytest
import torch

import optimi
from tests import reference

from tests.optimizer_test import (buffer, run_optimizer, gradient_release, cpu_dim1, cpu_dim2, cpu_dtype,
                                  cpu_ftype, gpu_dim1, gpu_dim2, gpu_dtype, gpu_ftype, gr_dim1,
                                  gr_dim2, gr_dtype, gr_ftype, optimizer_accumulation, gpu_device)



optimizers = {}

optimizers["sgd"] = ({'optim':torch.optim.SGD, 'kwargs':dict(lr=1e-3, momentum=0, dampening=0, weight_decay=0)},
                     {'optim':optimi.SGD, 'kwargs':dict(lr=1e-3, momentum=0, dampening=False, weight_decay=0)})

optimizers["sgd_mom"] = ({'optim':torch.optim.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=0, weight_decay=0)},
                         {'optim':optimi.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=False, weight_decay=0)})

optimizers["sgd_damp"] = ({'optim':torch.optim.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=0.9, weight_decay=0)},
                          {'optim':optimi.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=True, weight_decay=0, torch_init=True)})

optimizers["sgd_l2"] = ({'optim':torch.optim.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=0, weight_decay=1e-2)},
                        {'optim':optimi.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=False, weight_decay=1e-2, decouple_wd=False)})

optimizers["sgdw_dlr"] = ({'optim':reference.DecoupledSGDW, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=0.9, weight_decay=1e-5)},
                          {'optim':optimi.SGD, 'kwargs':dict(lr=1e-3, momentum=0.9, dampening=True, decouple_lr=True, weight_decay=1e-5, torch_init=True)})

optimizer_names = [key for key in optimizers.keys()]



cpu_values = list(product(cpu_dim1, cpu_dim2, cpu_dtype, optimizer_names, cpu_ftype))
cpu_names = ["dim1_{}_dim2_{}_dtype_{}_optim_{}{}".format(*vals) for vals in cpu_values]

@pytest.mark.cpu
@pytest.mark.sgd
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cpu_values, ids=cpu_names)
def test_optimizer_cpu(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str):
    run_optimizer(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device('cpu'), buffer)



cuda_values = list(product(gpu_dim1, gpu_dim2, gpu_dtype, optimizer_names, gpu_ftype))
cuda_names = ["dim1_{}_dim2_{}_dtype_{}_optim_{}{}".format(*vals) for vals in cuda_values]

@pytest.mark.gpu
@pytest.mark.sgd
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_optimizer_gpu(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str, gpu_device:str):
    run_optimizer(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device(gpu_device), buffer)



cuda_values = list(product(gr_dim1, gr_dim2, gr_dtype, optimizer_names, gr_ftype))
cuda_names = ["dim1_{}_dim2_{}_dtype_{}_optim_{}{}".format(*vals) for vals in cuda_values]

@pytest.mark.gpu
@pytest.mark.sgd
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_gradient_release(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str, gpu_device:str):
    gradient_release(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device(gpu_device),
                     framework_opt_step=torch.rand(1).item() > 0.5)


@pytest.mark.gpu
@pytest.mark.sgd
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_optimizer_accumulation(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str, gpu_device:str):
    if optim_name in ["sgd", "sgd_l2"]:
        pytest.skip("Skip tests for SGD and SGD with L2 weight decay.")
    # SGD will error out more often if iterations is the default of 80
    optimizer_accumulation(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device(gpu_device),
                           iterations=20, framework_opt_step=torch.rand(1).item() > 0.5)
