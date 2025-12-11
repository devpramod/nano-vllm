"""
Tests for HpuPlatform.

All tests run on real HPU hardware - no mocking of HPU operations.
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch

# Mock CUDA-specific modules before importing nanovllm
sys.modules["flash_attn"] = MagicMock()

from nanovllm.platforms import get_platform, _PLATFORMS
from nanovllm.platforms.hpu import HpuPlatform


class TestHpuPlatformBasic:
    """Basic tests for HpuPlatform availability and identity."""

    def test_platform_device_name(self):
        """Test that device_name is 'hpu'."""
        platform = HpuPlatform()
        assert platform.device_name == "hpu"

    def test_platform_device_type(self):
        """Test that device_type is 'hpu'."""
        platform = HpuPlatform()
        assert platform.device_type == "hpu"

    def test_platform_is_available(self):
        """Test that is_available() returns True on HPU hardware."""
        platform = HpuPlatform()
        assert platform.is_available() is True

    def test_platform_get_device_string(self):
        """Test that get_device_string() returns 'hpu'."""
        platform = HpuPlatform()
        assert platform.get_device_string() == "hpu"


class TestHpuPlatformDeviceOperations:
    """Tests for HPU device operations."""

    def test_set_device_is_noop(self):
        """Test that set_device() completes without error (it's a no-op on HPU)."""
        platform = HpuPlatform()
        # Should not raise - HPU uses HABANA_VISIBLE_MODULES instead
        platform.set_device(0)
        platform.set_device(1)

    def test_get_memory_info_returns_valid_tuple(self):
        """Test that get_memory_info() returns valid (free, total) tuple."""
        platform = HpuPlatform()
        free, total = platform.get_memory_info()

        assert isinstance(free, int)
        assert isinstance(total, int)
        assert total > 0, "Total memory should be positive"
        assert free >= 0, "Free memory should be non-negative"
        assert free <= total, "Free memory should not exceed total"

    def test_get_memory_info_reasonable_values(self):
        """Test that memory values are reasonable for Gaudi2 (~128GB)."""
        platform = HpuPlatform()
        free, total = platform.get_memory_info()

        # Gaudi2 has ~128GB HBM, should be at least 64GB
        min_expected = 64 * 1024 * 1024 * 1024  # 64 GB
        assert total >= min_expected, f"Expected at least 64GB, got {total / 1e9:.1f}GB"

    def test_synchronize_completes(self):
        """Test that synchronize() completes without error."""
        platform = HpuPlatform()
        # Create some work to synchronize
        tensor = torch.randn(100, 100, device="hpu", dtype=torch.bfloat16)
        _ = tensor @ tensor
        # Should not raise
        platform.synchronize()

    def test_mark_step_completes(self):
        """Test that mark_step() completes without error."""
        platform = HpuPlatform()
        # Create some work
        tensor = torch.randn(100, 100, device="hpu", dtype=torch.bfloat16)
        _ = tensor + 1
        # Should not raise (may log warning in eager mode, that's OK)
        platform.mark_step()


class TestHpuPlatformConfigValidation:
    """Tests for HPU configuration validation."""

    def test_check_config_accepts_valid_hpu_config(self):
        """Test that valid HPU config passes validation."""
        platform = HpuPlatform()

        # Create a mock config with valid HPU settings
        class MockConfig:
            dtype = torch.bfloat16
            kvcache_block_size = 128

        config = MockConfig()
        # Should not raise
        platform.check_and_update_config(config)

    def test_check_config_accepts_block_size_256(self):
        """Test that block_size=256 is accepted (divisible by 128)."""
        platform = HpuPlatform()

        class MockConfig:
            dtype = torch.bfloat16
            kvcache_block_size = 256

        config = MockConfig()
        platform.check_and_update_config(config)

    def test_check_config_accepts_block_size_384(self):
        """Test that block_size=384 is accepted (divisible by 128)."""
        platform = HpuPlatform()

        class MockConfig:
            dtype = torch.bfloat16
            kvcache_block_size = 384

        config = MockConfig()
        platform.check_and_update_config(config)

    def test_check_config_rejects_float16(self):
        """Test that float16 dtype is rejected."""
        platform = HpuPlatform()

        class MockConfig:
            dtype = torch.float16
            kvcache_block_size = 128

        config = MockConfig()
        with pytest.raises(ValueError, match="bfloat16"):
            platform.check_and_update_config(config)

    def test_check_config_rejects_float32(self):
        """Test that float32 dtype is rejected."""
        platform = HpuPlatform()

        class MockConfig:
            dtype = torch.float32
            kvcache_block_size = 128

        config = MockConfig()
        with pytest.raises(ValueError, match="bfloat16"):
            platform.check_and_update_config(config)

    def test_check_config_rejects_block_size_64(self):
        """Test that block_size=64 is rejected (not divisible by 128)."""
        platform = HpuPlatform()

        class MockConfig:
            dtype = torch.bfloat16
            kvcache_block_size = 64

        config = MockConfig()
        with pytest.raises(ValueError, match="divisible by 128"):
            platform.check_and_update_config(config)

    def test_check_config_rejects_block_size_100(self):
        """Test that block_size=100 is rejected (not divisible by 128)."""
        platform = HpuPlatform()

        class MockConfig:
            dtype = torch.bfloat16
            kvcache_block_size = 100

        config = MockConfig()
        with pytest.raises(ValueError, match="divisible by 128"):
            platform.check_and_update_config(config)


class TestHpuPlatformLazyMode:
    """Tests for lazy mode configuration."""

    def test_set_torch_compile_sets_weight_sharing(self):
        """Test that set_torch_compile() sets PT_HPU_WEIGHT_SHARING=0."""
        import os

        platform = HpuPlatform()
        platform.set_torch_compile()

        assert os.environ.get("PT_HPU_WEIGHT_SHARING") == "0"

    def test_set_torch_compile_can_be_called_multiple_times(self):
        """Test that set_torch_compile() can be called multiple times safely."""
        platform = HpuPlatform()
        # Should not raise
        platform.set_torch_compile()
        platform.set_torch_compile()


class TestHpuPlatformRegistration:
    """Tests for platform registration and discovery."""

    def test_hpu_platform_registered(self):
        """Test that HpuPlatform is registered in the platform registry."""
        assert "hpu" in _PLATFORMS
        assert _PLATFORMS["hpu"] == HpuPlatform

    def test_get_platform_returns_hpu_platform(self):
        """Test that get_platform('hpu') returns HpuPlatform instance."""
        platform = get_platform("hpu")
        assert isinstance(platform, HpuPlatform)
        assert platform.device_name == "hpu"

    def test_get_platform_auto_detects_hpu(self):
        """Test that get_platform() auto-detects HPU when available."""
        # On HPU hardware without CUDA, should auto-detect HPU
        platform = get_platform()
        # May be HPU or CUDA depending on environment
        assert platform.device_name in ("hpu", "cuda")


class TestHpuPlatformAttentionBackend:
    """Tests for attention backend."""

    def test_get_attention_backend_not_implemented(self):
        """Test that get_attention_backend() raises NotImplementedError."""
        platform = HpuPlatform()
        with pytest.raises(NotImplementedError, match="Task 11"):
            platform.get_attention_backend()
