import gc
import inspect
import json
import logging
import os
import time
import warnings
from collections.abc import Callable
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import torch._logging
import transformers
import typer
from optimi import SGD, Adam, AdamW, Adan, Lion, RAdam, Ranger, StableAdamW, param_groups_weight_decay
from optimi.optimizer import OptimiOptimizer
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    filesize,
)
from rich.table import Column
from rich.text import Text
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, PreTrainedModel
from typer import Option

logging.getLogger("torch._logging._internal").setLevel(logging.ERROR)

# Suppress torch.compile profiler warnings
torch._logging.set_logs(all=logging.ERROR)

# Alternative methods to suppress warnings:
# Method 1: Direct logger suppression
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

# Method 2: Environment variable approach (can also be set in shell)
os.environ["TORCH_LOGS"] = "-all"

# Method 3: Suppress specific warning categories
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*Profiler function.*will be ignored.*")

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

# ruff: noqa: D101, D102, D103, E501
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, pretty_exceptions_show_locals=False)

console = Console()


# from https://github.com/Textualize/rich/discussions/2035#discussioncomment-3516405
class RateColumn(ProgressColumn):
    """Renders human readable processing rate."""

    def __init__(self, label: str = "it/sec", table_column: Column | None = None):
        super().__init__(table_column)
        self.label = label

    def render(self, task: Task) -> Text:
        """Render the speed in iterations per second."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.data.speed")
        unit, suffix = filesize.pick_unit_and_suffix(int(speed), ["", "×10³", "×10⁶", "×10⁹", "×10¹²"], 1000)
        data_speed = speed / unit
        return Text(f"{data_speed:.2f}{suffix} {self.label}", style="progress.data.speed")


# modified from https://github.com/thomasbrandon/mish-cuda/blob/master/test/perftest.py
def scale_time(val: float, spec: str = "#0.4G"):
    "Scale fractional second `time` values and return formatted to `spec`."
    if val == 0:
        return "-"
    PREFIXES = np.array([c for c in "yzafpnµm kMGTPEZY"])
    exp = np.int8(np.log10(np.abs(val)) // 3 * 3 * np.sign(val))
    val /= 10.0**exp
    prefix = PREFIXES[exp // 3 + len(PREFIXES) // 2].strip()
    return f"{val:{spec}} {prefix}s"


class OptimizerEnum(str, Enum):
    optimizer_class: type[torch.optim.Optimizer | OptimiOptimizer]

    def __new__(cls, value: str, optimizer_class: torch.optim.Optimizer | OptimiOptimizer):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.optimizer_class = optimizer_class
        return obj

    @property
    def init_params(self):
        return set(inspect.signature(self.optimizer_class.__init__).parameters.keys())


class Optimizer(OptimizerEnum):
    torch_adam = ("torch_adam", torch.optim.Adam)
    torch_adamw = ("torch_adamw", torch.optim.AdamW)
    torch_sgd = ("torch_sgd", torch.optim.SGD)
    adam = ("optimi_adam", Adam)
    adamw = ("optimi_adamw", AdamW)
    adan = ("optimi_adan", Adan)
    lion = ("optimi_lion", Lion)
    radam = ("optimi_radam", RAdam)
    ranger = ("optimi_ranger", Ranger)
    sgd = ("optimi_sgd", SGD)
    stableadamw = ("optimi_stableadamw", StableAdamW)


class ModelType(str, Enum):
    causal_lm = "causal_lm"
    masked_lm = "masked_lm"
    vision = "vision"


class Precision(str, Enum):
    bf16 = "bf16"
    fp32 = "fp32"


def get_progress(detailed: bool = False) -> Progress:
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.remaining]Steps: {task.completed}/{task.total}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        RateColumn(label="steps/sec"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    if detailed:
        return Progress(*columns, console=console)
    else:
        cols = columns[:-3] + [columns[-2]]
        return Progress(*cols, console=console)


def can_compile(optimizer_class: torch.optim.Optimizer | OptimiOptimizer) -> bool:
    return issubclass(optimizer_class, torch.optim.Optimizer) and not issubclass(optimizer_class, OptimiOptimizer)


def load_model(
    model_name: str,
    trust_remote_code: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> PreTrainedModel:
    try:
        config = AutoConfig.from_pretrained(model_name)
        if "architectures" in config:
            # Create model directly on target device for fast initialization
            with torch.device(device):
                if any("forcausal" in arch.lower() for arch in config.architectures):
                    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code, torch_dtype=dtype)
                elif any("formasked" in arch.lower() for arch in config.architectures):
                    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=trust_remote_code, torch_dtype=dtype)
                else:
                    raise ValueError(f"Model {model_name} has unsupported architecture: {config.architectures}")

            return model
        else:
            raise ValueError(f"Model {model_name} does not have a valid architecture in its config.")
    except Exception as e:
        console.print(f"[red]Error creating model {model_name}: {e}[/red]")
        raise typer.Exit(code=1)


def setup(
    model_name: str,
    optimizer: Optimizer,
    weight_decay: float,
    foreach: bool,
    fused: bool,
    triton: bool,
    compiled: bool,
    gradient_release: bool,
    exclude_bias_norm: bool,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    kahan_sum: bool | None = None,
    trust_remote_code: bool = False,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, Callable | None]:
    # Load the model
    model = load_model(model_name, trust_remote_code=trust_remote_code, device=device, dtype=dtype)

    # Select optimizer
    optimizer_class = optimizer.optimizer_class
    optimizer_params = {
        "foreach": foreach,
        "fused": fused,
        "triton": triton,
        "weight_decay": weight_decay,
        "gradient_release": gradient_release,
        "kahan_sum": kahan_sum,
    }

    # disable fused/foreach for torch.optim when compiling torch optimizers
    if compiled and can_compile(optimizer_class):
        optimizer_params["fused"] = None
        optimizer_params["foreach"] = None
    elif compiled:
        raise ValueError(f"Only PyTorch optimizers can be compiled with torch.compile. Selected optimizer: {optimizer.value}")

    optimizer_params = {k: v for k, v in optimizer_params.items() if k in optimizer.init_params}

    if exclude_bias_norm and weight_decay > 0:
        optimizer = optimizer_class(param_groups_weight_decay(model, weight_decay=weight_decay), lr=1e-5, **optimizer_params)
    else:
        optimizer = optimizer_class(model.parameters(), lr=1e-5, **optimizer_params)

    # compile step only for torch.optim builtins (exclude OptimiOptimizer)
    if compiled and can_compile(optimizer_class):

        @torch.compile(fullgraph=False)
        def compiled_step():
            optimizer.step()

        return model, optimizer, compiled_step
    # compiled optimizer not supported for this optimizer type
    return model, optimizer, None


def run_benchmark(
    model_name: str = "meta-llama/Llama-3.2-1B",
    optimizer: Optimizer = Optimizer.adamw,
    precision: Precision = Precision.fp32,
    device: str | None = None,
    num_steps: int = 100,
    weight_decay: float = 0.0,
    foreach: bool = False,
    fused: bool = False,
    compiled: bool = False,
    gradient_release: bool = False,
    triton: bool = False,
    exclude_bias_norm: bool = False,
    kahan_sum: bool | None = None,
    warmup_steps: int = 10,
    progress: Progress | None = None,
    verbose: bool = True,
    trust_remote_code: bool = False,
):
    assert warmup_steps > 0, "warmup_steps must be greater than 0"
    assert num_steps > warmup_steps, "num_steps must be greater than warmup_steps"

    # Determine backend name
    suffix = " compiled" if compiled else ""
    active_backends = []
    if foreach:
        active_backends.append("foreach")
    if fused:
        active_backends.append("fused")
    if triton:
        active_backends.append("triton")
    backend_name = "/".join(active_backends) if active_backends else "torch"
    backend_display = f"{backend_name}{suffix}"

    console.print(f"[blue]Running optimizer benchmark for {model_name} with {optimizer.value} optimizer ({backend_display})[/blue]")

    # Determine device and dtype before model creation
    if torch.cuda.is_available():
        if device is None:
            device = "cuda"
        synchronize = torch.cuda.synchronize
        empty_cache = torch.cuda.empty_cache
        reset_peak_memory_stats = torch.cuda.reset_peak_memory_stats
        max_memory_allocated = torch.cuda.max_memory_allocated
    elif torch.xpu.is_available():
        if device is None:
            device = "xpu"
        synchronize = torch.xpu.synchronize
        empty_cache = torch.xpu.empty_cache
        reset_peak_memory_stats = torch.xpu.reset_peak_memory_stats
        max_memory_allocated = torch.xpu.max_memory_allocated
    else:
        raise ValueError(f"No suitable {device=} found. Please ensure you have a CUDA, ROCm, or XPU device available.")

    # Set dtype based on precision
    if precision == Precision.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    model, optimizer, compiled_step = setup(
        model_name=model_name,
        optimizer=optimizer,
        weight_decay=weight_decay,
        foreach=foreach,
        fused=fused,
        compiled=compiled,
        triton=triton,
        gradient_release=gradient_release,
        exclude_bias_norm=exclude_bias_norm,
        device=device,
        dtype=dtype,
        kahan_sum=kahan_sum,
        trust_remote_code=trust_remote_code,
    )

    device = torch.device(device)
    model.to(device, dtype=dtype)
    model.train()

    optimizer_times = []
    total_steps = num_steps + warmup_steps

    use_progress = progress is not None
    if use_progress:
        task = progress.add_task("[cyan]Benchmarking", total=total_steps)

    for step in range(total_steps):
        if step == 0:
            empty_cache()
            reset_peak_memory_stats(device)

        # Generate random gradients for all parameters with requires_grad=True
        # This simulates the backward pass without computing forward/backward
        for param in model.parameters():
            if param.requires_grad:
                if param.grad is None:
                    param.grad = torch.randn_like(param, device=device, dtype=dtype)
                else:
                    param.grad.data = torch.randn_like(param, device=device, dtype=dtype)

        # skip the warmup_steps so we don't measure warmup and compile time
        if step > warmup_steps:
            synchronize()
            start_time = time.perf_counter()

        if compiled:
            compiled_step()
        else:
            optimizer.step()

        if step > warmup_steps:
            synchronize()
            optimizer_times.append(time.perf_counter() - start_time)

        # both optimizers use the same zero_grad method, so we don't time it
        optimizer.zero_grad()

        if use_progress:
            progress.update(task, advance=1)

    optimizer_times = np.array(optimizer_times)

    results = {
        "opt_time_mean": optimizer_times.mean(),
        "opt_time_median": np.median(optimizer_times),
        "opt_time_90th": np.percentile(optimizer_times, 90),
        "total_time": np.sum(optimizer_times),
    }

    max_memory = max_memory_allocated(device=device)
    results["max_allocated_memory"] = filesize.decimal(max_memory)

    console.print(f"  Average Optimizer Time: {scale_time(results['opt_time_mean'])}")
    if verbose:
        console.print(f"  Median Optimizer Time: {scale_time(results['opt_time_median'])}")
        console.print(f"  90th Percentile Optimizer Time: {scale_time(results['opt_time_90th'])}")
        console.print(f"  Total Time (excluding warmup): {scale_time(results['total_time'])}")
        if "max_allocated_memory" in results:
            console.print(f"  Max Allocated Memory: {results['max_allocated_memory']}")

    model.to("cpu")
    model = None

    return results


@app.command(help="Benchmark a single optimizer on a single model")
def individual_optimizer(
    model: Annotated[str, Option(help="Causal or masked language model from Hugging Face transformers")] = "Qwen/Qwen3-0.6B",
    optimizer: Annotated[Optimizer, Option(help="Optimizer to benchmark")] = Optimizer.adamw,
    precision: Annotated[Precision, Option(help="Precision to benchmark. Use fp32 for AMP, bf16 for low precision")] = Precision.fp32,
    num_steps: Annotated[int, Option(help="Number of optimization steps to benchmark")] = 100,
    weight_decay: Annotated[float, Option(help="Weight decay")] = 0.0,
    foreach: Annotated[bool, Option(help="Use foreach implementation")] = False,
    fused: Annotated[bool, Option(help="Use fused implementation. Only valid for PyTorch optimizers")] = False,
    triton: Annotated[bool, Option(help="Use triton implementation. Only valid for optimi optimizers")] = False,
    compiled: Annotated[bool, Option(help="Use torch compile implementation. Only valid for PyTorch optimizers")] = False,
    gradient_release: Annotated[bool, Option(help="Use gradient release. Only valid for optimi optimizers")] = False,
    exclude_bias_norm: Annotated[bool, Option(help="Exclude bias and normalization layers from weight decay")] = False,
    kahan_sum: Annotated[bool | None, Option(help="Use Kahan summation. Defaults to true for optimi optimizers in low precision")] = None,
    device: Annotated[str | None, Option(help="Device to use. Defaults to 'cuda' or 'xpu' if available.")] = None,
    warmup_steps: Annotated[int, Option(help="Don't measure time for the first N steps. Must be greater than 0.")] = 10,
    trust_remote_code: Annotated[bool, Option(help="Trust remote model code")] = False,
):  # fmt: skip
    inner_progress = get_progress(detailed=True)
    progress_group = Group(inner_progress)
    progress_panel = Panel.fit(progress_group, title="Benchmark Progress", border_style="green", padding=(1, 1))

    with Live(progress_panel, console=console):
        run_benchmark(
            model_name=model,
            optimizer=optimizer,
            precision=precision,
            num_steps=num_steps,
            weight_decay=weight_decay,
            foreach=foreach,
            fused=fused,
            compiled=compiled,
            gradient_release=gradient_release,
            triton=triton,
            exclude_bias_norm=exclude_bias_norm,
            kahan_sum=kahan_sum,
            warmup_steps=warmup_steps,
            progress=inner_progress,
            device=device,
            trust_remote_code=trust_remote_code,
        )


@app.command(help="Benchmark multiple models and optimizers across foreach, fused, compiled, and triton backends.")
def benchmark_optimizers(
    models: Annotated[list[str], Option(help="Causal or masked language models from Hugging Face transformers")] = ["Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B"],
    optimizers: Annotated[list[Optimizer], Option(help="Optimizers to benchmark")] = [Optimizer.torch_adamw, Optimizer.adamw],
    precision: Annotated[Precision, Option(help="Precision to benchmark. Use fp32 to bench mixed precision, bf16 for low precision")] = Precision.bf16,
    num_steps: Annotated[int, Option(help="Number of optimization steps")] = 100,
    weight_decay: Annotated[float, Option(help="Weight decay")] = 1e-2,
    gradient_release: Annotated[bool, Option(help="Use gradient release. Only valid for optimi optimizers")] = False,
    exclude_bias_norm: Annotated[bool, Option(help="Exclude bias and norm from weight decay")] = True,
    kahan_sum: Annotated[bool | None, Option(help="Use Kahan summation. Defaults to true for optimi optimizers in low precision")] = None,
    device: Annotated[str | None, Option(help="Device to use. Defaults to 'cuda' or 'xpu' if available")] = None,
    warmup_steps: Annotated[int, Option(help="Number of warmup steps before timing starts")] = 10,
    results_file: Annotated[Path, Option(help="File to save results to. Appends to existing file")] = Path("benchmark_results.json"),
    verbose: Annotated[bool, Option(help="Verbose output")] = True,
    sleep: Annotated[int, Option(help="Sleep time between runs in seconds")] = 10,
    trust_remote_code: Annotated[bool, Option(help="Trust remote model code")] = False,
):  # fmt: skip
    progress_models = get_progress()
    progress_optimizers = get_progress()
    progress_backends = get_progress()
    inner_progress = get_progress(detailed=True)

    model_task = progress_models.add_task("Models", total=len(models))
    optimizer_task = progress_optimizers.add_task("Optimizers", total=len(optimizers))
    backend_task = progress_backends.add_task("Backends", total=0)

    progress_group = Group(progress_models, progress_optimizers, progress_backends, inner_progress)
    progress_panel = Panel.fit(progress_group, title="Benchmark Progress", border_style="green", padding=(1, 1))

    with Live(progress_panel, console=console):
        for model_name in models:
            progress_models.update(model_task, description=f"[cyan]Model: {model_name}")
            progress_optimizers.update(optimizer_task, total=len(optimizers), completed=0, description="Optimizers")

            for optimizer_choice in optimizers:
                supported_compile = can_compile(optimizer_choice.optimizer_class)
                supported = optimizer_choice.init_params
                domains = {
                    "foreach": [False, True] if "foreach" in supported else [False],
                    "fused": [False, True] if "fused" in supported else [False],
                    "triton": [False, True] if "triton" in supported else [False],
                    "compiled": [False, True] if supported_compile else [False],
                }
                keys = list(domains.keys())
                combos = [dict(zip(keys, vals)) for vals in product(*domains.values()) if sum(vals) <= 1]

                progress_optimizers.update(optimizer_task, description=f"[cyan]Optimizer: {optimizer_choice.value}")
                progress_backends.update(backend_task, total=len(combos), completed=0, description="[cyan]Backends")

                for flags in combos:
                    # display backend name
                    suffix = " compiled" if flags["compiled"] else ""
                    active = [k for k, v in flags.items() if v and k != "compiled"]
                    name = "/".join(active) if active else "torch"
                    desc = f"[cyan]Backend: {name}{suffix}"
                    progress_backends.update(backend_task, description=desc)
                    # clear inner tasks
                    for t in list(inner_progress.tasks):
                        inner_progress.remove_task(t.id)
                    # run benchmark
                    results = run_benchmark(
                        model_name=model_name,
                        optimizer=optimizer_choice,
                        precision=precision,
                        num_steps=num_steps,
                        weight_decay=weight_decay,
                        foreach=flags["foreach"],
                        fused=flags["fused"],
                        gradient_release=gradient_release,
                        triton=flags["triton"],
                        compiled=flags["compiled"],
                        exclude_bias_norm=exclude_bias_norm,
                        kahan_sum=kahan_sum,
                        warmup_steps=warmup_steps,
                        progress=inner_progress,
                        device=device,
                        verbose=verbose,
                        trust_remote_code=trust_remote_code,
                    )

                    entry = {
                        "model_name": model_name,
                        "optimizer": optimizer_choice.value,
                        **flags,
                        "precision": precision.value,
                    }
                    entry.update(results)

                    if results_file.exists():
                        try:
                            entries = json.loads(results_file.read_text())
                        except json.JSONDecodeError:
                            entries = []
                    else:
                        entries = []
                    entries.append(entry)
                    results_file.write_text(json.dumps(entries, indent=4))

                    torch.cuda.empty_cache()
                    gc.collect()

                    progress_backends.update(backend_task, advance=1)
                    if verbose and sleep > 0:
                        console.print(f"[blue]Sleeping for {sleep} seconds before next benchmark")
                        time.sleep(sleep)

                progress_optimizers.update(optimizer_task, advance=1)
            progress_models.update(model_task, advance=1)


@app.command(help="Estimate memory required for a model and optimizer")
def estimate_memory(
    model: Annotated[str, Option(help="Causal or masked language model from Hugging Face transformers")] = "Qwen/Qwen3-0.6B",
    optimizer: Annotated[Optimizer, Option(help="Optimizer to estimate memory for")] = Optimizer.adamw,
    precision: Annotated[Precision, Option(help="Precision to estimate (fp32 or bf16)")] = Precision.fp32,
    kahan_sum: Annotated[bool | None, Option(help="Use Kahan summation. Defaults to true for optimi optimizers in low precision")] = None,
    trust_remote_code: Annotated[bool, Option(help="Trust remote model code")] = False,
):
    """Estimate memory required for a model and optimizer."""
    from rich import print

    # Set dtype based on precision
    if precision == Precision.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()

    # Load model on CPU to avoid OOM
    model_name = model
    model = load_model(model, trust_remote_code=trust_remote_code, device="meta", dtype=dtype)
    total_params = sum(p.numel() for p in model.parameters())
    weights_mem = total_params * bytes_per_param

    # Determine optimizer buffer count
    # Dynamically determine buffer count using a dummy parameter to trigger actual state slots
    dummy = torch.nn.Parameter(torch.zeros(64, dtype=dtype, device="cpu", requires_grad=True))
    opt_cls = optimizer.optimizer_class
    init_params = inspect.signature(opt_cls.__init__).parameters
    opt_kwargs: dict = {}
    if "kahan_sum" in init_params:
        opt_kwargs["kahan_sum"] = kahan_sum
    opt = opt_cls([dummy], lr=0.0, **opt_kwargs)
    dummy.grad = torch.zeros_like(dummy)
    with torch.no_grad():
        opt.step()
    state = opt.state_dict().get("state", {})
    # buffer_count: count tensors that match dummy's shape
    buffer_count = max(
        (sum(1 for buf in slot.values() if isinstance(buf, torch.Tensor) and buf.shape == dummy.shape) for slot in state.values()),
        default=1,
    )

    optimizer_mem = total_params * bytes_per_param * buffer_count
    total_mem = weights_mem + optimizer_mem + weights_mem

    def format_mem(nbytes):
        for unit in ["B", "KB", "MB", "GB"]:
            if nbytes < 1024:
                return f"{nbytes:.2f} {unit}"
            nbytes /= 1024
        return f"{nbytes:.2f} TB"

    print(
        "\nEstimated memory required for loading the model and optimizer plus scratch for calculating the optimizer step."
        "\nExcludes activation memory, distributed buffers, and forward/backward scratch.\n"
    )
    print(f"Model: {model_name}")
    print(f"Precision: {precision} ({bytes_per_param} bytes per param)")
    print(f"Optimizer: {optimizer.value} ({buffer_count} buffer{'s' if buffer_count > 1 else ''} per param)")
    print(f"Parameters: {total_params:,}")
    print("Memory:")
    print(f"  Weights: {format_mem(weights_mem)}")
    print(f"  Gradients: {format_mem(weights_mem)}")
    print(f"  Optimizer State: {format_mem(optimizer_mem)}")
    print(f"  Total: {format_mem(total_mem)}")
    print(f"Estimated allocated memory: {format_mem(total_mem * 1.075)} - {format_mem(total_mem * 1.3)}\n")
    return total_mem


if __name__ == "__main__":
    app()
