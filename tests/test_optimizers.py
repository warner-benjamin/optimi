import pytest
import torch
from _pytest.mark.structures import ParameterSet


from .config import Backend, DeviceType, OptTest, OptTestType, discover_tests
from .runner import run_test

DEVICE_PARAMS = [
    pytest.param(DeviceType.cpu, marks=pytest.mark.cpu, id=DeviceType.cpu.value),
    pytest.param(DeviceType.gpu, marks=pytest.mark.gpu, id=DeviceType.gpu.value),
]
DTYPE_PARAMS = [
    pytest.param(torch.float32, marks=pytest.mark.float32, id="float32"),
    pytest.param(torch.bfloat16, marks=pytest.mark.bfloat16, id="bfloat16"),
]
BACKEND_PARAMS = [
    pytest.param(Backend.torch, marks=pytest.mark.torch, id=Backend.torch.value),
    pytest.param(Backend.foreach, marks=pytest.mark.foreach, id=Backend.foreach.value),
    pytest.param(Backend.triton, marks=pytest.mark.triton, id=Backend.triton.value),
]

# Attach per-optimizer marks so users can -m adam, -m sgd, etc.
OPTIMIZERS = [pytest.param(c, id=c.name, marks=getattr(pytest.mark, c.optimizer_name)) for c in discover_tests()]

# Full dimensions: CPU -> (64,64), (64,128); GPU -> (256,256), (256,512), (256,1024), (256,2048)
FULL_DIMS = [
    pytest.param((DeviceType.cpu, (64, 64)), id="cpu-64x64"),
    pytest.param((DeviceType.cpu, (64, 128)), id="cpu-64x128"),
    pytest.param((DeviceType.gpu, (256, 256)), id="gpu-256x256"),
    pytest.param((DeviceType.gpu, (256, 512)), id="gpu-256x512"),
    pytest.param((DeviceType.gpu, (256, 1024)), id="gpu-256x1024"),
    pytest.param((DeviceType.gpu, (256, 2048)), id="gpu-256x2048"),
]

# Gradient release and accumulation dims: (128,256) and (128,1024)
SUBSET_DIMS = [
    pytest.param((128, 256), id="gr-128x256"),
    pytest.param((128, 1024), id="gr-128x1024"),
]


def _should_skip(test_type: OptTestType, opttest: OptTest, device_type: DeviceType, dtype: torch.dtype, backend: Backend) -> bool:
    # 1. Hardware availability
    if not device_type.is_available():
        return True

    # 2. Backend support for hardware
    if not backend.is_supported(device_type):
        return True

    # 3. Explicit per-opttest skip
    if test_type in set(opttest.skip_tests):
        return True

    # 4. Respect per-test dtype constraints if provided
    if opttest.only_dtypes and dtype not in opttest.only_dtypes:
        return True

    # 5. Gradient release and accumulation are GPU-only tests
    if test_type in (OptTestType.gradient_release, OptTestType.accumulation) and device_type == DeviceType.cpu:
        return True

    # 6. bfloat16 is not supported on MPS
    if (
        device_type == DeviceType.gpu
        and dtype == torch.bfloat16
        and not (torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available()))
        and (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    ):
        return True

    # 7. Skip bfloat16 on CPU for most optimizers; allow anyadam exception via opttest.any_precision
    if device_type == DeviceType.cpu and dtype == torch.bfloat16 and not opttest.any_precision:
        return True

    # 8. Skip foreach for gradient release and accumulation tests
    if test_type != OptTestType.normal and backend == Backend.foreach:
        return True

    return False


def _param_value(param: ParameterSet) -> object:
    return param.values[0]


def _param_id(param: ParameterSet) -> str:
    return param.id or str(param.values[0])


def _build_params(test_type: OptTestType) -> list[ParameterSet]:
    if test_type == OptTestType.normal:
        device_params = DEVICE_PARAMS
        dtype_params = DTYPE_PARAMS
        dims_params = FULL_DIMS
    else:
        device_params = [pytest.param(DeviceType.gpu, marks=pytest.mark.gpu, id=DeviceType.gpu.value)]
        dtype_params = [pytest.param(torch.float32, marks=pytest.mark.float32, id="float32")]
        dims_params = SUBSET_DIMS

    params: list[ParameterSet] = []
    for opt_param in OPTIMIZERS:
        for device_param in device_params:
            for dtype_param in dtype_params:
                for backend_param in BACKEND_PARAMS:
                    for dims_param in dims_params:
                        if test_type == OptTestType.normal and _param_value(dims_param)[0] != _param_value(device_param):
                            continue
                        if _should_skip(
                            test_type,
                            _param_value(opt_param),
                            _param_value(device_param),
                            _param_value(dtype_param),
                            _param_value(backend_param),
                        ):
                            continue
                        param_id = "-".join(
                            [
                                _param_id(dims_param),
                                _param_id(backend_param),
                                _param_id(dtype_param),
                                _param_id(device_param),
                                _param_id(opt_param),
                            ]
                        )
                        params.append(
                            pytest.param(
                                _param_value(opt_param),
                                _param_value(device_param),
                                _param_value(dtype_param),
                                _param_value(backend_param),
                                _param_value(dims_param),
                                id=param_id,
                                marks=list(opt_param.marks + device_param.marks + dtype_param.marks + backend_param.marks),
                            )
                        )
    return params


@pytest.mark.parametrize("opttest, device_type, dtype, backend, dims_spec", _build_params(OptTestType.normal))
def test_normal(opttest, device_type, dtype, backend, dims_spec, gpu_device):
    _, dims = dims_spec
    device = torch.device(gpu_device if device_type == DeviceType.gpu else "cpu")
    run_test(opttest, device, dtype, backend, OptTestType.normal, dims=dims)


@pytest.mark.parametrize("opttest, device_type, dtype, backend, dims", _build_params(OptTestType.gradient_release))
def test_gradient_release(opttest, device_type, dtype, backend, dims, gpu_device):
    run_test(opttest, torch.device(gpu_device), dtype, backend, OptTestType.gradient_release, dims=dims)


@pytest.mark.parametrize("opttest, device_type, dtype, backend, dims", _build_params(OptTestType.accumulation))
def test_accumulation(opttest, device_type, dtype, backend, dims, gpu_device):
    run_test(opttest, torch.device(gpu_device), dtype, backend, OptTestType.accumulation, dims=dims)
