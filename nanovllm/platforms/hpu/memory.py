"""
HPU memory profiling utilities.

Provides HpuMemoryProfiler context manager for measuring memory consumption
on Intel Gaudi accelerators, following vllm-gaudi's HabanaMemoryProfiler pattern.
"""

import gc
import os
from functools import lru_cache
from typing import Optional

import torch


@lru_cache(maxsize=1)
def is_fake_hpu() -> bool:
    """
    Check if running with fake HPU (for testing without hardware).

    Returns:
        True if VLLM_USE_FAKE_HPU environment variable is set.
    """
    return os.environ.get("VLLM_USE_FAKE_HPU", "0") != "0"


def format_bytes(size: int) -> str:
    """
    Convert bytes to human-readable format.

    Args:
        size: Size in bytes.

    Returns:
        Human-readable string (e.g., "1.5 GiB").
    """
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PiB"


class HpuMemoryProfiler:
    """
    Context manager for measuring memory consumption on HPU.

    Follows vllm-gaudi's HabanaMemoryProfiler pattern. Uses gc.collect()
    and torch.hpu.synchronize() to ensure accurate measurements in lazy mode.

    Example:
        with HpuMemoryProfiler() as profiler:
            # ... allocate tensors ...
            torch.hpu.synchronize()
        print(profiler.get_summary_string())

    See: vllm-gaudi/vllm_gaudi/extension/profiler.py
    """

    def __init__(self, device: Optional[int] = None):
        """
        Initialize the memory profiler.

        Args:
            device: HPU device index (currently unused, for API compatibility).
        """
        self.device = device
        self.initial_memory: int = 0
        self.final_memory: int = 0
        self.consumed_memory: int = 0

    @staticmethod
    def current_device_memory_usage() -> int:
        """
        Get current device memory usage in bytes.

        Returns:
            Used memory (total - free) in bytes, or 0 if fake HPU.
        """
        if is_fake_hpu():
            return 0
        free, total = torch.hpu.mem_get_info()
        return total - free

    @staticmethod
    def current_free_device_memory() -> int:
        """
        Get current free device memory in bytes.

        Returns:
            Free memory in bytes, or 0 if fake HPU.
        """
        if is_fake_hpu():
            return 0
        free, _ = torch.hpu.mem_get_info()
        return free

    @staticmethod
    def total_device_memory() -> int:
        """
        Get total device memory in bytes.

        Returns:
            Total memory in bytes, or 0 if fake HPU.
        """
        if is_fake_hpu():
            return 0
        _, total = torch.hpu.mem_get_info()
        return total

    def __enter__(self) -> "HpuMemoryProfiler":
        """
        Start memory profiling.

        Performs garbage collection and captures initial memory state.
        """
        gc.collect()
        self.initial_memory = self.current_device_memory_usage()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        End memory profiling.

        Synchronizes HPU, performs garbage collection, and computes
        consumed memory as delta between final and initial states.
        """
        # Ensure all lazy operations complete before measuring
        if not is_fake_hpu():
            torch.hpu.synchronize()
        gc.collect()
        self.final_memory = self.current_device_memory_usage()
        self.consumed_memory = self.final_memory - self.initial_memory

    def get_summary_string(self) -> str:
        """
        Get human-readable summary of memory consumption.

        Returns:
            Formatted string like "1.5 GiB consumed (10.2/32.0 GiB used)".

        Raises:
            RuntimeError: If called before exiting the context manager.
        """
        if self.final_memory == 0 and self.consumed_memory == 0 and not is_fake_hpu():
            raise RuntimeError(
                "get_summary_string() called before profiling completed. "
                "Use within 'with' block or after exiting context."
            )

        total = self.total_device_memory()
        return (
            f"{format_bytes(self.consumed_memory)} consumed "
            f"({format_bytes(self.final_memory)}/{format_bytes(total)} used)"
        )
