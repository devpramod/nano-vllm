"""
pytest configuration and fixtures for nano-vllm test suite.

Provides fixtures and markers for testing on both CUDA and HPU platforms.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from typing import Optional


def is_hpu_available() -> bool:
    """Check if HPU (Gaudi) hardware is available."""
    try:
        import habana_frameworks.torch as htorch
        return torch.hpu.is_available()
    except (ImportError, AttributeError):
        return False


def is_cuda_available() -> bool:
    """Check if CUDA hardware is available."""
    return torch.cuda.is_available()


def get_device_count(device_type: str) -> int:
    """Get number of available devices for given type."""
    if device_type == "hpu":
        try:
            return torch.hpu.device_count()
        except (ImportError, AttributeError):
            return 0
    elif device_type == "cuda":
        return torch.cuda.device_count()
    return 0


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def hpu_available() -> bool:
    """Fixture that returns True if HPU is available."""
    return is_hpu_available()


@pytest.fixture
def cuda_available() -> bool:
    """Fixture that returns True if CUDA is available."""
    return is_cuda_available()


@pytest.fixture
def device_type(hpu_available: bool, cuda_available: bool) -> str:
    """Fixture that returns the available device type ('hpu', 'cuda', or 'cpu')."""
    if hpu_available:
        return "hpu"
    elif cuda_available:
        return "cuda"
    return "cpu"


@pytest.fixture
def mock_hpu_device():
    """
    Fixture that mocks HPU device for testing without hardware.

    Use this fixture when writing tests that should work without HPU hardware.
    """
    mock_htorch = MagicMock()
    mock_htorch.core.mark_step = MagicMock()

    with patch.dict('sys.modules', {'habana_frameworks': MagicMock(),
                                     'habana_frameworks.torch': mock_htorch}):
        with patch('torch.hpu.is_available', return_value=True):
            with patch('torch.hpu.device_count', return_value=1):
                with patch('torch.hpu.set_device', return_value=None):
                    with patch('torch.hpu.synchronize', return_value=None):
                        with patch('torch.hpu.empty_cache', return_value=None):
                            with patch('torch.hpu.mem_get_info', return_value=(10*1024**3, 32*1024**3)):
                                yield mock_htorch


@pytest.fixture
def mock_cuda_device():
    """
    Fixture that mocks CUDA device for testing without hardware.
    """
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.cuda.device_count', return_value=1):
            with patch('torch.cuda.set_device', return_value=None):
                with patch('torch.cuda.synchronize', return_value=None):
                    with patch('torch.cuda.empty_cache', return_value=None):
                        with patch('torch.cuda.mem_get_info', return_value=(10*1024**3, 24*1024**3)):
                            yield


@pytest.fixture
def sample_config():
    """
    Fixture that returns a sample configuration dict for testing.
    """
    return {
        "model": "Qwen/Qwen2-0.5B",
        "dtype": "bfloat16",
        "max_seq_len": 2048,
        "kvcache_block_size": 128,
        "tensor_parallel_size": 1,
    }


@pytest.fixture
def sample_prompts():
    """
    Fixture that returns sample prompts for testing.
    """
    return [
        "Hello, world!",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]


# ============================================================================
# Markers and Hooks
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "hpu: mark test as requiring HPU hardware"
    )
    config.addinivalue_line(
        "markers", "multi_hpu: mark test as requiring multiple HPUs"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests marked with @pytest.mark.hpu if HPU is not available.
    """
    if is_hpu_available():
        # HPU is available, don't skip hpu tests
        return

    skip_hpu = pytest.mark.skip(reason="HPU not available")
    skip_multi_hpu = pytest.mark.skip(reason="Multiple HPUs not available")

    for item in items:
        if "hpu" in item.keywords:
            item.add_marker(skip_hpu)
        if "multi_hpu" in item.keywords:
            item.add_marker(skip_multi_hpu)


# ============================================================================
# Helper Functions for Tests
# ============================================================================

def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    msg: Optional[str] = None
):
    """
    Assert that two tensors are close within tolerance.

    Useful for comparing outputs between CUDA and HPU implementations.
    """
    # Move tensors to CPU for comparison
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
