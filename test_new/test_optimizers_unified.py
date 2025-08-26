import pytest
import torch
from optimi.utils import MIN_TORCH_2_6

from .cases import discover_cases
from .runners import run_accumulation, run_correctness, run_gradient_release

CASES = tuple(discover_cases())

DEVICE_PARAMS = [
    pytest.param("cpu", marks=pytest.mark.cpu, id="cpu"),
    pytest.param("gpu", marks=pytest.mark.gpu, id="gpu"),
]
DTYPE_PARAMS = [
    pytest.param(torch.float32, marks=pytest.mark.float32, id="float32"),
    pytest.param(torch.bfloat16, marks=pytest.mark.bfloat16, id="bfloat16"),
]
BACKEND_PARAMS = [
    pytest.param("torch", marks=pytest.mark.torch, id="torch"),
    pytest.param("triton", marks=pytest.mark.triton, id="triton"),
]

# Attach per-optimizer marks so users can -m adam, -m sgd, etc.
OPTIM_PARAMS = [pytest.param(c, id=c.name, marks=getattr(pytest.mark, c.optimizer_name)) for c in CASES]

# Dimension parameter spaces (match legacy tests)
# Correctness dims: CPU -> (64,64), (64,128); GPU -> (256,256), (256,512), (256,1024), (256,2048)
CORRECTNESS_DIMS = [
    pytest.param(("cpu", (64, 64)), id="cpu-64x64"),
    pytest.param(("cpu", (64, 128)), id="cpu-64x128"),
    pytest.param(("gpu", (256, 256)), id="gpu-256x256"),
    pytest.param(("gpu", (256, 512)), id="gpu-256x512"),
    pytest.param(("gpu", (256, 1024)), id="gpu-256x1024"),
    pytest.param(("gpu", (256, 2048)), id="gpu-256x2048"),
]

# Gradient release and accumulation dims (GPU-only): (128,256) and (128,1024)
GR_DIMS = [
    pytest.param((128, 256), id="gr-128x256"),
    pytest.param((128, 1024), id="gr-128x1024"),
]


def _should_skip(test_type, case, device_type, dtype, backend) -> bool:
    # Explicit per-case skip
    if test_type in set(case.skip_tests):
        return True

    # Respect per-test dtype constraints if provided
    if case.only_dtypes and dtype not in case.only_dtypes:
        return True

    # Skip triton on CPU
    if backend == "triton" and device_type == "cpu":
        return True

    # Triton requires torch >= 2.6
    if backend == "triton" and not MIN_TORCH_2_6:
        return True

    # GPU availability
    if device_type == "gpu" and not (
        torch.cuda.is_available()
        or (hasattr(torch, "xpu") and torch.xpu.is_available())
        or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    ):
        return True

    # Gradient release and accumulation are GPU-only tests
    if test_type in ["gradient_release", "accumulation"] and device_type == "cpu":
        return True

    # Skip bfloat16 on CPU for most optimizers; allow anyadam-style exceptions via case.any_precision if needed
    if device_type == "cpu" and dtype == torch.bfloat16 and not case.any_precision:
        return True

    return False


@pytest.mark.parametrize("case", OPTIM_PARAMS)
@pytest.mark.parametrize("device_type", DEVICE_PARAMS)
@pytest.mark.parametrize("dtype", DTYPE_PARAMS)
@pytest.mark.parametrize("backend", BACKEND_PARAMS)
@pytest.mark.parametrize("dims_spec", CORRECTNESS_DIMS)
def test_correctness(case, device_type, dtype, backend, dims_spec, gpu_device):
    if _should_skip("correctness", case, device_type, dtype, backend):
        pytest.skip()
    dims_device, dims = dims_spec
    if dims_device != device_type:
        pytest.skip()
    device = torch.device(gpu_device if device_type == "gpu" else "cpu")
    run_correctness(case, device, dtype, backend, dims=dims)


@pytest.mark.parametrize("case", OPTIM_PARAMS)
@pytest.mark.parametrize("device_type", [pytest.param("gpu", marks=pytest.mark.gpu, id="gpu")])
@pytest.mark.parametrize("dtype", [pytest.param(torch.float32, marks=pytest.mark.float32, id="float32")])
@pytest.mark.parametrize("backend", BACKEND_PARAMS)
@pytest.mark.parametrize("dims", GR_DIMS)
def test_gradient_release(case, device_type, dtype, backend, dims, gpu_device):
    if _should_skip("gradient_release", case, device_type, dtype, backend):
        pytest.skip()
    run_gradient_release(case, torch.device(gpu_device), dtype, backend, dims=dims)


@pytest.mark.parametrize("case", OPTIM_PARAMS)
@pytest.mark.parametrize("device_type", [pytest.param("gpu", marks=pytest.mark.gpu, id="gpu")])
@pytest.mark.parametrize("dtype", [pytest.param(torch.float32, marks=pytest.mark.float32, id="float32")])
@pytest.mark.parametrize("backend", BACKEND_PARAMS)
@pytest.mark.parametrize("dims", GR_DIMS)
def test_accumulation(case, device_type, dtype, backend, dims, gpu_device):
    if _should_skip("accumulation", case, device_type, dtype, backend):
        pytest.skip()
    run_accumulation(case, torch.device(gpu_device), dtype, backend, dims=dims)
