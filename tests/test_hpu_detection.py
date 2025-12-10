"""
Basic HPU detection and functionality tests.

All tests require real HPU hardware.
"""

import pytest
import torch
import habana_frameworks.torch as htorch


class TestHPUDetection:
    """Tests for HPU hardware detection and basic operations."""

    def test_hpu_is_available(self):
        """Test that HPU is available."""
        assert torch.hpu.is_available(), "HPU should be available"

    def test_hpu_device_count(self):
        """Test that at least one HPU device is detected."""
        count = torch.hpu.device_count()
        assert count >= 1, f"Expected at least 1 HPU, got {count}"

    def test_hpu_can_create_tensor(self, hpu_device):
        """Test that tensors can be created on HPU."""
        tensor = torch.randn(10, 10, device="hpu", dtype=torch.bfloat16)
        assert tensor.device.type == "hpu"
        assert tensor.dtype == torch.bfloat16

    def test_hpu_basic_matmul(self, hpu_device):
        """Test that matrix multiplication works on HPU."""
        a = torch.randn(100, 100, device="hpu", dtype=torch.bfloat16)
        b = torch.randn(100, 100, device="hpu", dtype=torch.bfloat16)
        c = torch.matmul(a, b)
        assert c.shape == (100, 100)
        assert c.device.type == "hpu"

    def test_hpu_memory_info(self):
        """Test that HPU memory info can be retrieved."""
        free, total = torch.hpu.mem_get_info()
        assert total > 0, "Total memory should be positive"
        assert free >= 0, "Free memory should be non-negative"
        assert free <= total, "Free memory should not exceed total"

    def test_hpu_synchronize(self, hpu_device):
        """Test that HPU synchronization works."""
        tensor = torch.randn(10, 10, device="hpu", dtype=torch.bfloat16)
        result = tensor + 1
        torch.hpu.synchronize()
        assert result is not None

    def test_mark_step(self, hpu_device):
        """Test that mark_step can be called (lazy mode sync point)."""
        tensor = torch.randn(10, 10, device="hpu", dtype=torch.bfloat16)
        result = tensor * 2
        htorch.core.mark_step()
        assert result is not None

    def test_hpu_memory_stats(self, hpu_device):
        """Test that memory stats can be retrieved."""
        tensor = torch.randn(1000, 1000, device="hpu", dtype=torch.bfloat16)
        # Check memory is being used
        free_before, total = torch.hpu.mem_get_info()
        del tensor
        torch.hpu.synchronize()
        free_after, _ = torch.hpu.mem_get_info()
        assert total > 0


class TestHPUOperations:
    """Tests for various HPU tensor operations."""

    def test_softmax(self, hpu_device):
        """Test softmax on HPU."""
        x = torch.randn(32, 128, device="hpu", dtype=torch.bfloat16)
        result = torch.softmax(x, dim=-1)
        assert result.shape == x.shape
        # Check softmax sums to 1
        sums = result.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-2)

    def test_layer_norm(self, hpu_device):
        """Test layer normalization on HPU."""
        x = torch.randn(32, 128, device="hpu", dtype=torch.bfloat16)
        ln = torch.nn.LayerNorm(128, device="hpu", dtype=torch.bfloat16)
        result = ln(x)
        assert result.shape == x.shape

    def test_linear(self, hpu_device):
        """Test linear layer on HPU."""
        x = torch.randn(32, 128, device="hpu", dtype=torch.bfloat16)
        linear = torch.nn.Linear(128, 256, device="hpu", dtype=torch.bfloat16)
        result = linear(x)
        assert result.shape == (32, 256)

    def test_silu_activation(self, hpu_device):
        """Test SiLU activation on HPU."""
        x = torch.randn(32, 128, device="hpu", dtype=torch.bfloat16)
        result = torch.nn.functional.silu(x)
        assert result.shape == x.shape

    def test_embedding(self, hpu_device):
        """Test embedding lookup on HPU."""
        vocab_size = 1000
        embed_dim = 128
        emb = torch.nn.Embedding(vocab_size, embed_dim, device="hpu", dtype=torch.bfloat16)
        indices = torch.randint(0, vocab_size, (32, 16), device="hpu")
        result = emb(indices)
        assert result.shape == (32, 16, embed_dim)


@pytest.mark.multi_hpu
class TestMultiHPU:
    """Tests requiring multiple HPUs.

    Note: On Gaudi, multi-HPU is controlled via HABANA_VISIBLE_MODULES env var,
    not torch.hpu.set_device(). Each process sees only one device.
    True multi-HPU testing requires distributed process groups.
    """

    def test_multi_hpu_available(self, multi_hpu):
        """Test that multiple HPUs are detected."""
        assert multi_hpu >= 2

    def test_device_count_matches(self, multi_hpu):
        """Test device count matches expected."""
        count = torch.hpu.device_count()
        assert count == multi_hpu
