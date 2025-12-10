"""
Tests for Config with HPU support.

These tests validate the configuration system including:
- New HPU-specific fields (device_type, distributed_backend, lazy_mode, use_fused_sdpa, dtype)
- HPU constraint validation (block size % 128, bfloat16 only)
- Auto-detection for device_type and distributed_backend
- Backward compatibility with CUDA configs
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch

# Mock CUDA-specific modules before importing nanovllm
sys.modules["flash_attn"] = MagicMock()

from nanovllm.config import Config  # noqa: E402


# Small model for testing - HuggingFace will download/cache automatically
MODEL_ID = "facebook/opt-125m"


@pytest.fixture
def model_id():
    """Fixture providing a model ID for Config tests."""
    return MODEL_ID


class TestConfigHPUFields:
    """Tests for new HPU-specific Config fields."""

    def test_device_type_accepts_hpu(self, model_id):
        """Test that device_type='hpu' is accepted."""
        config = Config(model=model_id, device_type="hpu")
        assert config.device_type == "hpu"

    def test_distributed_backend_auto_selects_hccl_for_hpu(self, model_id):
        """Test that HPU auto-selects HCCL backend."""
        config = Config(model=model_id, device_type="hpu")
        assert config.distributed_backend == "hccl"

    def test_lazy_mode_defaults_true(self, model_id):
        """Test lazy_mode defaults to True."""
        config = Config(model=model_id, device_type="hpu")
        assert config.lazy_mode is True

    def test_use_fused_sdpa_defaults_true(self, model_id):
        """Test use_fused_sdpa defaults to True."""
        config = Config(model=model_id, device_type="hpu")
        assert config.use_fused_sdpa is True

    def test_dtype_defaults_bfloat16(self, model_id):
        """Test dtype defaults to bfloat16."""
        config = Config(model=model_id, device_type="hpu")
        assert config.dtype == torch.bfloat16

    def test_get_dtype_method(self, model_id):
        """Test get_dtype() helper method."""
        config = Config(model=model_id, device_type="hpu")
        assert config.get_dtype() == torch.bfloat16


class TestHPUConstraints:
    """Tests for HPU-specific constraint validation."""

    def test_hpu_block_size_defaults_to_128(self, model_id):
        """Test HPU defaults to block size 128."""
        config = Config(model=model_id, device_type="hpu")
        assert config.kvcache_block_size == 128

    def test_hpu_accepts_block_size_256(self, model_id):
        """Test HPU accepts block size 256 (divisible by 128)."""
        config = Config(model=model_id, device_type="hpu", kvcache_block_size=256)
        assert config.kvcache_block_size == 256

    def test_hpu_accepts_block_size_384(self, model_id):
        """Test HPU accepts block size 384 (divisible by 128)."""
        config = Config(model=model_id, device_type="hpu", kvcache_block_size=384)
        assert config.kvcache_block_size == 384

    def test_hpu_rejects_block_size_64(self, model_id):
        """Test HPU rejects block size 64."""
        with pytest.raises(ValueError, match="divisible by 128"):
            Config(model=model_id, device_type="hpu", kvcache_block_size=64)

    def test_hpu_rejects_block_size_100(self, model_id):
        """Test HPU rejects block size 100."""
        with pytest.raises(ValueError, match="divisible by 128"):
            Config(model=model_id, device_type="hpu", kvcache_block_size=100)

    def test_hpu_rejects_float16(self, model_id):
        """Test HPU rejects float16 dtype."""
        with pytest.raises(ValueError, match="bfloat16"):
            Config(model=model_id, device_type="hpu", dtype=torch.float16)

    def test_hpu_rejects_float32(self, model_id):
        """Test HPU rejects float32 dtype."""
        with pytest.raises(ValueError, match="bfloat16"):
            Config(model=model_id, device_type="hpu", dtype=torch.float32)


class TestAutoDetection:
    """Tests for device and backend auto-detection."""

    def test_explicit_hpu_overrides_auto(self, model_id):
        """Test explicit device_type='hpu' works."""
        config = Config(model=model_id, device_type="hpu")
        assert config.device_type == "hpu"
        assert config.distributed_backend == "hccl"

    def test_explicit_backend_preserved(self, model_id):
        """Test explicit backend setting is preserved."""
        config = Config(model=model_id, device_type="hpu", distributed_backend="hccl")
        assert config.distributed_backend == "hccl"

    def test_existing_fields_preserved(self, model_id):
        """Test that existing Config fields work correctly."""
        config = Config(
            model=model_id,
            device_type="hpu",
            max_num_batched_tokens=8192,
            max_num_seqs=256,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=2,
            enforce_eager=True,
        )
        assert config.max_num_batched_tokens == 8192
        assert config.max_num_seqs == 256
        assert config.gpu_memory_utilization == 0.85
        assert config.tensor_parallel_size == 2
        assert config.enforce_eager is True


class TestValidationErrorMessages:
    """Tests ensuring validation errors are clear and helpful."""

    def test_hpu_block_size_error_includes_value(self, model_id):
        """Test HPU block size error shows the invalid value."""
        with pytest.raises(ValueError) as exc_info:
            Config(model=model_id, device_type="hpu", kvcache_block_size=64)

        error_msg = str(exc_info.value)
        assert "128" in error_msg  # Shows required alignment
        assert "64" in error_msg   # Shows actual value

    def test_hpu_dtype_error_mentions_bfloat16(self, model_id):
        """Test HPU dtype error mentions the required dtype."""
        with pytest.raises(ValueError) as exc_info:
            Config(model=model_id, device_type="hpu", dtype=torch.float32)

        error_msg = str(exc_info.value)
        assert "bfloat16" in error_msg
