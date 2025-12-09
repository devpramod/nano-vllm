"""
Basic HPU detection tests.

These tests verify that the HPU hardware is correctly detected and accessible.
"""

import pytest
import torch


class TestHPUDetection:
    """Tests for HPU hardware detection."""

    @pytest.mark.hpu
    def test_hpu_is_available(self):
        """Test that HPU is available when running on Gaudi hardware."""
        assert torch.hpu.is_available(), "HPU should be available"

    @pytest.mark.hpu
    def test_hpu_device_count(self):
        """Test that at least one HPU device is detected."""
        count = torch.hpu.device_count()
        assert count >= 1, f"Expected at least 1 HPU, got {count}"

    @pytest.mark.hpu
    def test_hpu_can_create_tensor(self):
        """Test that tensors can be created on HPU."""
        tensor = torch.randn(10, 10, device="hpu", dtype=torch.bfloat16)
        assert tensor.device.type == "hpu"
        assert tensor.dtype == torch.bfloat16

    @pytest.mark.hpu
    def test_hpu_basic_operation(self):
        """Test that basic tensor operations work on HPU."""
        a = torch.randn(100, 100, device="hpu", dtype=torch.bfloat16)
        b = torch.randn(100, 100, device="hpu", dtype=torch.bfloat16)
        c = torch.matmul(a, b)
        assert c.shape == (100, 100)
        assert c.device.type == "hpu"

    @pytest.mark.hpu
    def test_hpu_memory_info(self):
        """Test that HPU memory info can be retrieved."""
        free, total = torch.hpu.mem_get_info()
        assert total > 0, "Total memory should be positive"
        assert free >= 0, "Free memory should be non-negative"
        assert free <= total, "Free memory should not exceed total"

    @pytest.mark.hpu
    def test_hpu_synchronize(self):
        """Test that HPU synchronization works."""
        tensor = torch.randn(10, 10, device="hpu", dtype=torch.bfloat16)
        result = tensor + 1
        torch.hpu.synchronize()  # Should not raise
        assert result is not None

    @pytest.mark.hpu
    def test_habana_frameworks_import(self):
        """Test that habana_frameworks can be imported."""
        try:
            import habana_frameworks.torch as htorch
            assert htorch is not None
        except ImportError:
            pytest.fail("Failed to import habana_frameworks.torch")

    @pytest.mark.hpu
    def test_mark_step(self):
        """Test that mark_step can be called (lazy mode sync point)."""
        import habana_frameworks.torch as htorch
        tensor = torch.randn(10, 10, device="hpu", dtype=torch.bfloat16)
        result = tensor * 2
        htorch.core.mark_step()  # Should not raise
        assert result is not None


class TestMockedHPU:
    """Tests that work with mocked HPU (no hardware required)."""

    def test_mock_hpu_available(self, mock_hpu_device):
        """Test that mocked HPU reports as available."""
        assert torch.hpu.is_available()

    def test_mock_hpu_device_count(self, mock_hpu_device):
        """Test that mocked HPU reports device count."""
        assert torch.hpu.device_count() == 1

    def test_mock_hpu_memory_info(self, mock_hpu_device):
        """Test that mocked HPU returns memory info."""
        free, total = torch.hpu.mem_get_info()
        assert total == 32 * 1024**3  # 32GB mocked
        assert free == 10 * 1024**3   # 10GB free mocked


class TestDeviceFixture:
    """Tests for the device_type fixture."""

    def test_device_type_fixture(self, device_type):
        """Test that device_type fixture returns valid device."""
        assert device_type in ("hpu", "cuda", "cpu")

    def test_hpu_available_fixture(self, hpu_available):
        """Test that hpu_available fixture returns boolean."""
        assert isinstance(hpu_available, bool)

    def test_cuda_available_fixture(self, cuda_available):
        """Test that cuda_available fixture returns boolean."""
        assert isinstance(cuda_available, bool)
