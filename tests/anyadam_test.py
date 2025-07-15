from itertools import product

import pytest
import torch

import optimi
from tests import reference

from tests.optimizer_test import buffer, run_optimizer, cpu_dim1, cpu_dim2, cpu_dtype, cpu_ftype, gpu_dim1, gpu_dim2, gpu_dtype, gpu_ftype, gpu_device



optimizers = {}

optimizers["any_adam"] = ({'optim':reference.AnyPrecisionAdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0)},
                           {'optim':optimi.Adam, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=0, kahan_sum=True)})

optimizers["any_adamw"] = ({'optim':reference.AnyPrecisionAdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2)},
                           {'optim':optimi.AdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2, kahan_sum=True)})

optimizers["any_adamw_dlr"] = ({'optim':reference.AnyPrecisionAdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-2)},
                               {'optim':optimi.AdamW, 'kwargs':dict(lr=1e-3, betas=(0.9, 0.99), eps=1e-6, weight_decay=1e-5, decouple_lr=True, kahan_sum=True)})

optimizer_names = [key for key in optimizers.keys()]



cpu_dtype = [torch.bfloat16]
cpu_values = list(product(cpu_dim1, cpu_dim2, cpu_dtype, optimizer_names, cpu_ftype))
cpu_names = ["dim1_{}_dim2_{}_dtype_{}_optim_{}{}".format(*vals) for vals in cpu_values]

@pytest.mark.cpu
@pytest.mark.adam
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cpu_values, ids=cpu_names)
def test_optimizer_cpu(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str):
    run_optimizer(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device('cpu'), buffer, any_precision=True)



gpu_dtype = [torch.bfloat16]
cuda_values = list(product(gpu_dim1, gpu_dim2, gpu_dtype, optimizer_names, gpu_ftype))
cuda_names = ["dim1_{}_dim2_{}_dtype_{}_optim_{}{}".format(*vals) for vals in cuda_values]

@pytest.mark.gpu
@pytest.mark.adam
@pytest.mark.parametrize("dim1, dim2, dtype, optim_name, ftype", cuda_values, ids=cuda_names)
def test_optimizer_gpu(dim1:int, dim2:int, dtype:torch.dtype, optim_name:str, ftype:str, gpu_device:str):
    run_optimizer(optimizers, dim1, dim2, dtype, optim_name, ftype, torch.device(gpu_device), buffer, any_precision=True)