"""
Critical tests for HPU communicator - verifies mark_step() is called before collectives.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

# Mock flash_attn before any nanovllm imports
sys.modules["flash_attn"] = MagicMock()
sys.modules["flash_attn.flash_attn_interface"] = MagicMock()

from nanovllm.distributed.hpu_communicator import HpuCommunicator


class TestHpuCommunicatorMarkStep:
    """Critical: Verify mark_step() is called BEFORE each collective operation."""

    def test_all_reduce_calls_mark_step_before_collective(self):
        """all_reduce calls mark_step() BEFORE dist.all_reduce()."""
        comm = HpuCommunicator(world_size=2, rank=0)
        tensor = torch.tensor([1.0, 2.0, 3.0])
        call_order = []

        with patch.object(comm, "_mark_step", side_effect=lambda: call_order.append("mark_step")):
            with patch("torch.distributed.all_reduce", side_effect=lambda t: call_order.append("all_reduce")):
                comm.all_reduce(tensor)

        assert call_order == ["mark_step", "all_reduce"], "mark_step must be called BEFORE all_reduce"

    def test_barrier_calls_mark_step_before_collective(self):
        """barrier calls mark_step() BEFORE dist.barrier()."""
        comm = HpuCommunicator(world_size=2, rank=0)
        call_order = []

        with patch.object(comm, "_mark_step", side_effect=lambda: call_order.append("mark_step")):
            with patch("torch.distributed.barrier", side_effect=lambda: call_order.append("barrier")):
                comm.barrier()

        assert call_order == ["mark_step", "barrier"], "mark_step must be called BEFORE barrier"

    def test_gather_calls_mark_step_before_collective(self):
        """gather calls mark_step() BEFORE dist.gather()."""
        comm = HpuCommunicator(world_size=2, rank=0)
        tensor = torch.tensor([1.0, 2.0])
        call_order = []

        with patch.object(comm, "_mark_step", side_effect=lambda: call_order.append("mark_step")):
            with patch("torch.distributed.gather", side_effect=lambda *args: call_order.append("gather")):
                comm.gather(tensor, dst=0)

        assert call_order == ["mark_step", "gather"], "mark_step must be called BEFORE gather"

    def test_all_gather_calls_mark_step_before_collective(self):
        """all_gather calls mark_step() BEFORE dist.all_gather_into_tensor()."""
        comm = HpuCommunicator(world_size=2, rank=0)
        tensor = torch.tensor([1.0, 2.0])
        call_order = []

        with patch.object(comm, "_mark_step", side_effect=lambda: call_order.append("mark_step")):
            with patch("torch.distributed.all_gather_into_tensor", side_effect=lambda *args: call_order.append("all_gather")):
                comm.all_gather(tensor, dim=-1)

        assert call_order == ["mark_step", "all_gather"], "mark_step must be called BEFORE all_gather"


class TestHpuPlatformIntegration:
    """Verify HpuPlatform returns HpuCommunicator."""

    def test_hpu_platform_returns_hpu_communicator(self):
        """HpuPlatform.get_communicator_cls() returns HpuCommunicator."""
        from nanovllm.platforms.hpu import HpuPlatform

        platform = HpuPlatform()
        comm_cls = platform.get_communicator_cls()
        assert comm_cls is HpuCommunicator
