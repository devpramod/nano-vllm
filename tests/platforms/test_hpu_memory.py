"""
Critical tests for HPU memory management.

Tests HpuMemoryProfiler context manager and HpuPlatform memory methods.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

# Mock flash_attn before any nanovllm imports
sys.modules["flash_attn"] = MagicMock()
sys.modules["flash_attn.flash_attn_interface"] = MagicMock()

from nanovllm.platforms.hpu.memory import (
    HpuMemoryProfiler,
    is_fake_hpu,
    format_bytes,
)
from nanovllm.platforms.hpu import HpuPlatform


class TestHpuMemoryProfiler:
    """Critical tests for HpuMemoryProfiler context manager."""

    def test_profiler_tracks_memory_delta(self):
        """Profiler computes consumed_memory as final - initial."""
        profiler = HpuMemoryProfiler()

        # Simulate __enter__ and __exit__
        with patch.object(
            HpuMemoryProfiler,
            "current_device_memory_usage",
            side_effect=[100, 350],  # initial=100, final=350
        ):
            with patch("torch.hpu.synchronize"):
                profiler.__enter__()
                profiler.__exit__(None, None, None)

        assert profiler.initial_memory == 100
        assert profiler.final_memory == 350
        assert profiler.consumed_memory == 250  # 350 - 100

    def test_profiler_calls_gc_collect(self):
        """Profiler calls gc.collect() in __enter__ and __exit__."""
        with patch("gc.collect") as mock_gc:
            with patch.object(HpuMemoryProfiler, "current_device_memory_usage", return_value=0):
                with patch("torch.hpu.synchronize"):
                    with patch("nanovllm.platforms.hpu.memory.is_fake_hpu", return_value=True):
                        profiler = HpuMemoryProfiler()
                        profiler.__enter__()
                        profiler.__exit__(None, None, None)

        assert mock_gc.call_count == 2  # Once in __enter__, once in __exit__

    def test_profiler_syncs_before_exit_measurement(self):
        """Profiler calls torch.hpu.synchronize() before final measurement."""
        call_order = []

        with patch("torch.hpu.synchronize", side_effect=lambda: call_order.append("sync")):
            with patch("gc.collect", side_effect=lambda: call_order.append("gc")):
                with patch.object(
                    HpuMemoryProfiler,
                    "current_device_memory_usage",
                    side_effect=lambda: call_order.append("mem") or 0,
                ):
                    with patch("nanovllm.platforms.hpu.memory.is_fake_hpu", return_value=False):
                        profiler = HpuMemoryProfiler()
                        profiler.__enter__()
                        call_order.clear()  # Only track __exit__
                        profiler.__exit__(None, None, None)

        # sync should come before gc and memory measurement
        assert call_order[0] == "sync"


class TestFormatBytes:
    """Tests for format_bytes utility."""

    def test_format_bytes_units(self):
        """format_bytes converts to appropriate units."""
        assert "B" in format_bytes(500)
        assert "KiB" in format_bytes(1024)
        assert "MiB" in format_bytes(1024 * 1024)
        assert "GiB" in format_bytes(1024 * 1024 * 1024)


class TestHpuPlatformMemoryMethods:
    """Critical tests for HpuPlatform memory methods."""

    def test_empty_cache_calls_gc_collect(self):
        """empty_cache() calls gc.collect()."""
        platform = HpuPlatform()

        with patch("gc.collect") as mock_gc:
            platform.empty_cache()
            mock_gc.assert_called_once()

    def test_reset_peak_memory_stats_calls_gc_collect(self):
        """reset_peak_memory_stats() calls gc.collect()."""
        platform = HpuPlatform()

        with patch("gc.collect") as mock_gc:
            platform.reset_peak_memory_stats()
            mock_gc.assert_called_once()

    def test_get_peak_and_current_memory_return_same(self):
        """get_peak_memory() and get_current_memory() return same value on HPU."""
        platform = HpuPlatform()

        with patch.object(HpuMemoryProfiler, "current_device_memory_usage", return_value=12345):
            peak = platform.get_peak_memory()
            current = platform.get_current_memory()

        assert peak == current == 12345

    def test_memory_methods_use_profiler_static_method(self):
        """Memory methods delegate to HpuMemoryProfiler.current_device_memory_usage()."""
        platform = HpuPlatform()

        with patch.object(HpuMemoryProfiler, "current_device_memory_usage") as mock_usage:
            mock_usage.return_value = 999
            result = platform.get_current_memory()

        mock_usage.assert_called_once()
        assert result == 999
