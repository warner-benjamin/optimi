"""Main test class with comprehensive test methods for the unified optimizer test framework.

This module provides the TestOptimizers class that implements all test types:
- test_optimizer_correctness: Validates optimizer correctness against reference implementations
- test_gradient_release: Tests gradient release functionality (GPU-only)
- test_optimizer_accumulation: Tests optimizer accumulation functionality (GPU-only)
"""

import io

import pytest
import torch
from optimi import prepare_for_gradient_release, remove_gradient_release
from optimi.utils import MIN_TORCH_2_6
from torch import Tensor

from .framework import OptimizerTest, ToleranceConfig, assert_most_approx_close
from .pytest_integration import (
    create_float32_only_dtypes,
    create_gpu_only_device_types,
    create_marked_backends,
    create_marked_device_types,
    create_marked_dtypes,
    create_marked_optimizer_tests,
)


class MLP(torch.nn.Module):
    """Simple MLP model for testing optimizer behavior."""

    def __init__(self, input_size: int, hidden_size: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.norm = torch.nn.LayerNorm(input_size, device=device, dtype=dtype)
        self.fc1 = torch.nn.Linear(input_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act = torch.nn.Mish()
        self.fc2 = torch.nn.Linear(hidden_size, 1, bias=False, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TestOptimizers:
    """Main test class for comprehensive optimizer testing."""

    @pytest.mark.parametrize("optimizer_test", create_marked_optimizer_tests())
    @pytest.mark.parametrize("device_type", create_marked_device_types())
    @pytest.mark.parametrize("dtype", create_marked_dtypes())
    @pytest.mark.parametrize("backend", create_marked_backends())
    def test_optimizer_correctness(
        self,
        optimizer_test: OptimizerTest,
        device_type: str,
        dtype: torch.dtype,
        backend: str,
        gpu_device: str,
    ) -> None:
        """Test optimizer correctness against reference implementation.

        Validates that the optimi optimizer produces results consistent with
        the reference PyTorch optimizer across different configurations.
        """
        # Skip test if conditions don't match
        if self._should_skip("correctness", optimizer_test, device_type, dtype, backend):
            pytest.skip(f"Skipping {optimizer_test.name} correctness test for {device_type}/{dtype}/{backend}")

        # Log a random seed for reproducibility while keeping randomness
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        print(f"[seed] correctness: {optimizer_test.name} {device_type}/{dtype}/{backend} -> {seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Determine actual device
        device = torch.device(gpu_device if device_type == "gpu" else "cpu")

        # Run the correctness test
        self._run_correctness_test(optimizer_test, device, dtype, backend)

    @pytest.mark.parametrize("optimizer_test", create_marked_optimizer_tests())
    @pytest.mark.parametrize("device_type", create_gpu_only_device_types())
    @pytest.mark.parametrize("dtype", create_float32_only_dtypes())
    @pytest.mark.parametrize("backend", create_marked_backends())
    def test_gradient_release(
        self,
        optimizer_test: OptimizerTest,
        device_type: str,
        dtype: torch.dtype,
        backend: str,
        gpu_device: str,
    ) -> None:
        """Test gradient release functionality (GPU only).

        Validates that gradient release produces consistent results with
        standard optimizer behavior while freeing memory during backprop.
        """
        # Skip test if conditions don't match
        if self._should_skip("gradient_release", optimizer_test, device_type, dtype, backend):
            pytest.skip(f"Skipping {optimizer_test.name} gradient_release test for {device_type}/{dtype}/{backend}")

        # Log a random seed for reproducibility while keeping randomness
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        print(f"[seed] gradient_release: {optimizer_test.name} {device_type}/{dtype}/{backend} -> {seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Determine actual device (always GPU for this test)
        device = torch.device(gpu_device)

        # Run the gradient release test
        self._run_gradient_release_test(optimizer_test, device, dtype, backend)

    @pytest.mark.parametrize("optimizer_test", create_marked_optimizer_tests())
    @pytest.mark.parametrize("device_type", create_gpu_only_device_types())
    @pytest.mark.parametrize("dtype", create_float32_only_dtypes())
    @pytest.mark.parametrize("backend", create_marked_backends())
    def test_optimizer_accumulation(
        self,
        optimizer_test: OptimizerTest,
        device_type: str,
        dtype: torch.dtype,
        backend: str,
        gpu_device: str,
    ) -> None:
        """Test optimizer accumulation functionality (GPU only).

        Validates that optimizer accumulation produces results consistent
        with gradient accumulation while being more memory efficient.
        """
        # Skip test if conditions don't match
        if self._should_skip("accumulation", optimizer_test, device_type, dtype, backend):
            pytest.skip(f"Skipping {optimizer_test.name} accumulation test for {device_type}/{dtype}/{backend}")

        # Determine actual device (always GPU for this test)
        device = torch.device(gpu_device)

        # Run the accumulation test
        self._run_accumulation_test(optimizer_test, device, dtype, backend)

    def _prepare_kwargs(self, optimizer_test: OptimizerTest, backend: str) -> tuple[dict, dict]:
        """Prepare reference and optimi kwargs including backend-specific flags."""
        reference_kwargs = optimizer_test.to_reference_kwargs()
        optimi_kwargs = optimizer_test.to_optimi_kwargs()
        if backend == "triton":
            optimi_kwargs["triton"] = True
        else:
            optimi_kwargs["foreach"] = False
            optimi_kwargs["triton"] = False
        return reference_kwargs, optimi_kwargs

    def _should_skip(
        self,
        test_type: str,
        optimizer_test: OptimizerTest,
        device_type: str,
        dtype: torch.dtype,
        backend: str,
    ) -> bool:
        """Comprehensive test skipping logic with all conditions.

        Args:
            test_type: Type of test ('correctness', 'gradient_release', 'accumulation')
            optimizer_test: The optimizer test configuration
            device_type: Device type ('cpu' or 'gpu')
            dtype: Data type (torch.float32 or torch.bfloat16)
            backend: Backend type ('torch' or 'triton')

        Returns:
            True if test should be skipped, False otherwise
        """
        # Check if test type is explicitly skipped
        if optimizer_test.should_skip_test(test_type):
            return True

        # Respect per-test dtype constraints if provided
        if getattr(optimizer_test, "only_dtypes", None):
            if dtype not in optimizer_test.only_dtypes:
                return True

        # Skip triton tests on CPU
        if backend == "triton" and device_type == "cpu":
            return True

        # Skip triton tests if PyTorch version is too old
        if backend == "triton" and not MIN_TORCH_2_6:
            return True

        # Skip GPU tests if no GPU is available
        if device_type == "gpu" and not (torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())):
            return True

        # Gradient release and accumulation are GPU-only tests
        if test_type in ["gradient_release", "accumulation"] and device_type == "cpu":
            return True

        # Skip bfloat16 on CPU for most optimizers (matches original test behavior)
        # Only anyadam tests bfloat16 on CPU for precision testing
        if device_type == "cpu" and dtype == torch.bfloat16 and not optimizer_test.name.startswith("anyadam"):
            return True

        return False

    def _run_correctness_test(
        self,
        optimizer_test: OptimizerTest,
        device: torch.device,
        dtype: torch.dtype,
        backend: str,
    ) -> None:
        """Core correctness test implementation.

        Creates two identical models, runs them with optimi and reference optimizers,
        and validates that they produce consistent results.
        """
        # Get test configuration
        iterations = optimizer_test.get_iterations_for_test("correctness")
        tolerance = optimizer_test.get_tolerance_for_dtype(dtype)

        # Determine model dimensions and error handling based on device
        if device.type == "cpu":
            dim1, dim2 = 64, 128
            batch_size = 1
            max_error_count = 2
        else:
            dim1, dim2 = 256, 512
            batch_size = 32
            max_error_count = 5

        # Set max_error_rate for bfloat16 like original tests
        max_error_rate = None
        if dtype == torch.bfloat16:
            max_error_rate = 0.01  # Allow 1% of values to be outside tolerance

        # Skip 1x1 tests
        if dim1 == 1 and dim2 == 1:
            pytest.skip("Skipping 1x1 optimizer test")

        # Create models
        m1 = MLP(dim1, dim2, device=device, dtype=dtype)
        m2 = MLP(dim1, dim2, device=device, dtype=dtype)
        m2.load_state_dict(m1.state_dict())

        # Convert model parameters to float for non-any_precision testing
        if not optimizer_test.any_precision and dtype != torch.float32:
            for p in m1.parameters():
                p.data = p.data.float()

        # Create optimizers
        reference_class = optimizer_test.reference_class
        reference_kwargs, optimi_kwargs = self._prepare_kwargs(optimizer_test, backend)

        reference_optimizer = reference_class(m1.parameters(), **reference_kwargs)
        optimi_optimizer = optimizer_test.optimi_class(m2.parameters(), **optimi_kwargs)

        # Training loop with state dict testing
        buffer = io.BytesIO()

        for i in range(iterations):
            # Generate training data
            input1 = torch.randn(batch_size, dim1, device=device, dtype=dtype)
            input2 = input1.detach().clone()
            target1 = torch.randn(batch_size, 1, device=device, dtype=dtype)
            target2 = target1.detach().clone()

            # Convert inputs to float for non-any_precision testing
            if not optimizer_test.any_precision and dtype != torch.float32:
                input1 = input1.float()
                target1 = target1.float()

            # Forward pass
            output1 = m1(input1)
            output2 = m2(input2)

            # Loss calculation
            loss1 = torch.nn.functional.mse_loss(output1, target1)
            loss2 = torch.nn.functional.mse_loss(output2, target2)

            # Backward pass
            loss1.backward()
            loss2.backward()

            # Optimizer step
            reference_optimizer.step()
            optimi_optimizer.step()

            # Zero gradients
            reference_optimizer.zero_grad()
            optimi_optimizer.zero_grad()

            # Compare model weights
            assert_most_approx_close(
                m1.fc1.weight,
                m2.fc1.weight,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
                max_error_count=max_error_count,
                max_error_rate=tolerance.max_error_rate,
                name="fc1: ",
            )
            assert_most_approx_close(
                m1.fc2.weight,
                m2.fc2.weight,
                atol=tolerance.atol,
                rtol=tolerance.rtol,
                max_error_count=max_error_count,
                max_error_rate=tolerance.max_error_rate,
                name="fc2: ",
            )

            # Test state_dict saving and loading periodically
            if i % max(1, iterations // 10) == 0 and i > 0:
                # Save optimizer state
                torch.save(optimi_optimizer.state_dict(), buffer)
                buffer.seek(0)
                # Load checkpoint
                ckpt = torch.load(buffer, weights_only=True)
                # Recreate optimizer and load its state
                optimi_optimizer = optimizer_test.optimi_class(m2.parameters(), **optimi_kwargs)
                optimi_optimizer.load_state_dict(ckpt)
                # Clear buffer
                buffer.seek(0)
                buffer.truncate(0)

                # Verify models are still aligned after state_dict loading
                assert_most_approx_close(
                    m1.fc1.weight,
                    m2.fc1.weight,
                    atol=tolerance.atol,
                    rtol=tolerance.rtol,
                    max_error_count=max_error_count,
                    max_error_rate=tolerance.max_error_rate,
                    name="fc1 after load: ",
                )
                assert_most_approx_close(
                    m1.fc2.weight,
                    m2.fc2.weight,
                    atol=tolerance.atol,
                    rtol=tolerance.rtol,
                    max_error_count=max_error_count,
                    max_error_rate=tolerance.max_error_rate,
                    name="fc2 after load: ",
                )

    def _run_gradient_release_test(
        self,
        optimizer_test: OptimizerTest,
        device: torch.device,
        dtype: torch.dtype,
        backend: str,
    ) -> None:
        """Core gradient release test implementation.

        Compares gradient release behavior with torch PyTorch optimizer hooks
        and regular optimizer behavior to ensure consistency.
        """

        def optimizer_hook(parameter) -> None:
            torch_optimizers[parameter].step()
            torch_optimizers[parameter].zero_grad()

        # Get test configuration
        iterations = optimizer_test.get_iterations_for_test("gradient_release")
        tolerance = optimizer_test.get_tolerance_for_dtype(dtype)

        # Set default tolerances for gradient release (slightly more lenient)
        if optimizer_test.custom_tolerances is None:
            if dtype == torch.float32:
                tolerance = ToleranceConfig(rtol=1e-5, atol=2e-6)
            elif dtype == torch.bfloat16:
                tolerance = ToleranceConfig(rtol=1e-2, atol=2e-3)

        # Since Lion & Adan can have noisy updates, allow up to 12 errors
        max_error_count = 12

        # Model dimensions for gradient release tests
        dim1, dim2 = 128, 256
        batch_size = 32

        # Create three identical models
        m1 = MLP(dim1, dim2, device=device, dtype=dtype)  # Regular optimizer
        m2 = MLP(dim1, dim2, device=device, dtype=dtype)  # PyTorch hooks
        m3 = MLP(dim1, dim2, device=device, dtype=dtype)  # Optimi gradient release
        m2.load_state_dict(m1.state_dict())
        m3.load_state_dict(m1.state_dict())

        # Create optimizers
        reference_class = optimizer_test.reference_class
        reference_kwargs, optimi_kwargs = self._prepare_kwargs(optimizer_test, backend)

        # Regular optimizer
        regular_optimizer = reference_class(m1.parameters(), **reference_kwargs)

        # PyTorch Method: taken from https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
        torch_optimizers = {p: reference_class([p], **reference_kwargs) for p in m2.parameters()}

        pytorch_hooks = []
        for p in m2.parameters():
            pytorch_hooks.append(p.register_post_accumulate_grad_hook(optimizer_hook))

        # Optimi Method with gradient release
        optimi_kwargs["gradient_release"] = True
        optimi_optimizer = optimizer_test.optimi_class(m3.parameters(), **optimi_kwargs)
        prepare_for_gradient_release(m3, optimi_optimizer)

        # Training loop
        for i in range(iterations):
            input1 = torch.randn(batch_size, dim1, device=device, dtype=dtype)
            input2 = input1.clone()
            input3 = input1.clone()
            target1 = torch.randn(batch_size, 1, device=device, dtype=dtype)
            target2 = target1.clone()
            target3 = target1.clone()

            output1 = m1(input1)
            output2 = m2(input2)
            output3 = m3(input3)

            loss1 = torch.nn.functional.mse_loss(output1, target1)
            loss2 = torch.nn.functional.mse_loss(output2, target2)
            loss3 = torch.nn.functional.mse_loss(output3, target3)

            loss1.backward()
            loss2.backward()
            loss3.backward()

            regular_optimizer.step()
            regular_optimizer.zero_grad()

            # Simulate framework optimizer step (randomly enabled)
            framework_opt_step = torch.rand(1).item() > 0.5
            if framework_opt_step:
                optimi_optimizer.step()
                optimi_optimizer.zero_grad()

            # Compare results
            assert_most_approx_close(
                m1.fc1.weight,
                m2.fc1.weight,
                rtol=tolerance.rtol,
                atol=tolerance.atol,
                max_error_count=max_error_count,
                name="PyTorch-PyTorch: ",
            )
            assert_most_approx_close(
                m1.fc2.weight,
                m2.fc2.weight,
                rtol=tolerance.rtol,
                atol=tolerance.atol,
                max_error_count=max_error_count,
                name="PyTorch-PyTorch: ",
            )
            assert_most_approx_close(
                m1.fc1.weight,
                m3.fc1.weight,
                rtol=tolerance.rtol,
                atol=tolerance.atol,
                max_error_count=max_error_count,
                name="PyTorch-Optimi: ",
            )
            assert_most_approx_close(
                m1.fc2.weight,
                m3.fc2.weight,
                rtol=tolerance.rtol,
                atol=tolerance.atol,
                max_error_count=max_error_count,
                name="PyTorch-Optimi: ",
            )

        # Cleanup
        for h in pytorch_hooks:
            h.remove()
        remove_gradient_release(m3)

    def _run_accumulation_test(
        self,
        optimizer_test: OptimizerTest,
        device: torch.device,
        dtype: torch.dtype,
        backend: str,
    ) -> None:
        """Core accumulation test implementation.

        Tests optimizer accumulation functionality which approximates gradient
        accumulation by accumulating directly into optimizer states.
        """
        # Get test configuration
        iterations = optimizer_test.get_iterations_for_test("accumulation")

        # Since optimizer accumulation approximates gradient accumulation,
        # the tolerances are high despite the low number of iterations
        max_error_rate = 0.035
        tolerance = ToleranceConfig(rtol=1e-2, atol=1e-2)

        # Model dimensions for accumulation tests
        dim1, dim2 = 128, 256
        batch_size = 32

        # Create two identical models
        m1 = MLP(dim1, dim2, device=device, dtype=dtype)  # Regular optimizer
        m2 = MLP(dim1, dim2, device=device, dtype=dtype)  # Optimi accumulation
        m2.load_state_dict(m1.state_dict())

        # Create optimizers
        reference_class = optimizer_test.reference_class
        reference_kwargs, optimi_kwargs = self._prepare_kwargs(optimizer_test, backend)

        # Regular optimizer
        regular_optimizer = reference_class(m1.parameters(), **reference_kwargs)

        # Optimi optimizer with gradient release for accumulation
        optimi_kwargs["gradient_release"] = True
        optimi_optimizer = optimizer_test.optimi_class(m2.parameters(), **optimi_kwargs)
        prepare_for_gradient_release(m2, optimi_optimizer)

        gradient_accumulation_steps = 4

        # Training loop
        for i in range(iterations):
            input1 = torch.randn(batch_size, dim1, device=device, dtype=dtype)
            input2 = input1.clone()
            target1 = torch.randn(batch_size, 1, device=device, dtype=dtype)
            target2 = target1.clone()

            # Set accumulation mode
            optimi_optimizer.optimizer_accumulation = (i + 1) % gradient_accumulation_steps != 0

            output1 = m1(input1)
            output2 = m2(input2)

            loss1 = torch.nn.functional.mse_loss(output1, target1)
            loss2 = torch.nn.functional.mse_loss(output2, target2)

            loss1.backward()
            loss2.backward()

            # Only step regular optimizer when not accumulating
            if not optimi_optimizer.optimizer_accumulation:
                regular_optimizer.step()
                regular_optimizer.zero_grad()

            # Simulate framework optimizer step (randomly enabled)
            framework_opt_step = torch.rand(1).item() > 0.5
            if framework_opt_step:
                optimi_optimizer.step()
                optimi_optimizer.zero_grad()

        # Unlike other tests, compare that the weights are in the same approximate range at the end of training
        assert_most_approx_close(m1.fc1.weight, m2.fc1.weight, rtol=tolerance.rtol, atol=tolerance.atol, max_error_rate=max_error_rate)
        assert_most_approx_close(m1.fc2.weight, m2.fc2.weight, rtol=tolerance.rtol, atol=tolerance.atol, max_error_rate=max_error_rate)

        # Cleanup
        remove_gradient_release(m2)
