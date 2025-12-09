"""
pytest configuration and fixtures for nano-vllm test suite.

All tests run on real HPU hardware - no mocking.
"""

import pytest
import torch
from typing import Optional

import habana_frameworks.torch as htorch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def hpu_device():
    """Fixture that sets up HPU device for testing."""
    assert torch.hpu.is_available(), "HPU hardware required"
    torch.hpu.set_device(0)
    yield
    torch.hpu.synchronize()


@pytest.fixture
def multi_hpu():
    """Fixture for multi-HPU tests."""
    count = torch.hpu.device_count()
    assert count >= 2, f"Multi-HPU tests require at least 2 HPUs, found {count}"
    yield count


@pytest.fixture
def sample_config():
    """Sample configuration for HPU testing."""
    return {
        "model": "Qwen/Qwen2-0.5B",
        "dtype": "bfloat16",
        "max_seq_len": 2048,
        "kvcache_block_size": 128,
        "tensor_parallel_size": 1,
    }


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "Hello, world!",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]


# ============================================================================
# Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "multi_hpu: mark test as requiring multiple HPUs"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    msg: Optional[str] = None
):
    """
    Assert that two tensors are close within tolerance.

    Uses bfloat16-appropriate tolerances by default.
    """
    actual_cpu = actual.detach().float().cpu()
    expected_cpu = expected.detach().float().cpu()

    if not torch.allclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol):
        diff = (actual_cpu - expected_cpu).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        error_msg = f"Tensors not close. Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)
