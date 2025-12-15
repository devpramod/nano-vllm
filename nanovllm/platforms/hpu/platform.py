"""
HPU platform implementation for Intel Gaudi accelerators.

This module provides the HpuPlatform class that implements the Platform
interface for Gaudi hardware, supporting lazy mode execution and
HPU-specific optimizations.
"""

import os
from typing import TYPE_CHECKING, Tuple, Type

import torch

from nanovllm.platforms.base import Platform

if TYPE_CHECKING:
    from nanovllm.config import Config

# Import Habana frameworks with graceful fallback
try:
    import habana_frameworks.torch as htorch

    _HABANA_AVAILABLE = True
except ImportError:
    htorch = None
    _HABANA_AVAILABLE = False


class HpuPlatform(Platform):
    """
    Platform implementation for Intel Gaudi (HPU) accelerators.

    Uses Habana's PyTorch integration with lazy mode execution
    and FusedSDPA for optimized attention.

    Key differences from CUDA:
    - Device selection via HABANA_VISIBLE_MODULES env var, not set_device()
    - Lazy mode requires mark_step() synchronization points
    - Only bfloat16 dtype supported
    - Block size must be divisible by 128
    """

    device_name = "hpu"
    device_type = "hpu"

    def __init__(self):
        """Initialize HPU platform."""
        super().__init__()

    def check_and_update_config(self, config: "Config") -> None:
        """
        Validate and update configuration for HPU platform.

        Enforces HPU-specific constraints:
        - dtype must be bfloat16
        - block_size must be divisible by 128

        Args:
            config: The Config object to validate and potentially modify.

        Raises:
            ValueError: If configuration is invalid for HPU.
        """
        # Enforce bfloat16 - HPU doesn't support float16/float32 efficiently
        if config.dtype != torch.bfloat16:
            raise ValueError(
                f"HPU only supports bfloat16 dtype, got {config.dtype}. "
                f"Set dtype=torch.bfloat16 for HPU."
            )

        # Enforce block size divisible by 128
        if config.kvcache_block_size % 128 != 0:
            raise ValueError(
                f"KV cache block size must be divisible by 128 for HPU, "
                f"got {config.kvcache_block_size}"
            )

    def get_attention_backend(self) -> Type:
        """
        Get the attention backend class for HPU.

        Returns:
            The HPU attention backend class using FusedSDPA.

        Raises:
            NotImplementedError: HPU attention backend not yet implemented.
        """
        # TODO: Implement in Task 11 (HPU Attention Implementation)
        raise NotImplementedError(
            "HPU attention backend not yet implemented. "
            "See Task 11: HPU Attention Implementation."
        )

    def set_device(self, rank: int) -> None:
        """
        Set the current HPU device.

        Note: On Gaudi, device selection is typically done via the
        HABANA_VISIBLE_MODULES environment variable rather than
        explicit set_device calls. This method is a no-op.

        Args:
            rank: The device index (ignored on HPU).
        """
        # HPU device selection is handled via HABANA_VISIBLE_MODULES
        # See: https://docs.habana.ai/en/latest/PyTorch/Reference/Runtime_Flags.html
        pass

    def get_memory_info(self) -> Tuple[int, int]:
        """
        Get memory information for the current HPU device.

        Returns:
            Tuple of (free_bytes, total_bytes).
        """
        return torch.hpu.mem_get_info()

    def synchronize(self) -> None:
        """
        Synchronize the current HPU device.

        Blocks until all HPU operations are complete.
        """
        torch.hpu.synchronize()

    def is_available(self) -> bool:
        """
        Check if HPU is available on this system.

        Returns:
            True if Gaudi hardware and drivers are available.
        """
        if not _HABANA_AVAILABLE:
            return False
        try:
            return torch.hpu.is_available()
        except AttributeError:
            return False

    # =========================================================================
    # HPU-specific methods (not in base Platform)
    # =========================================================================

    def mark_step(self) -> None:
        """
        Insert a synchronization point for lazy mode execution.

        In lazy mode, operations are accumulated and executed in batches.
        mark_step() triggers execution of accumulated operations and is
        required before:
        - Distributed collective operations
        - Host-device data transfers
        - Control flow decisions based on tensor values

        This is a no-op if Habana frameworks are not available.
        """
        if htorch is not None:
            htorch.core.mark_step()

    def set_torch_compile(self) -> None:
        """
        Configure PyTorch compilation settings for HPU.

        Sets environment variables required for lazy mode execution:
        - PT_HPU_WEIGHT_SHARING=0 (required for Gaudi2)
        - PT_HPU_ENABLE_LAZY_COLLECTIVES=true (for multi-HPU with lazy mode)
        - Disables torch.compile in lazy mode (not supported)
        """
        # Required for Gaudi2
        os.environ["PT_HPU_WEIGHT_SHARING"] = "0"

        # Check if running in lazy mode
        is_lazy = True
        if htorch is not None:
            try:
                is_lazy = htorch.utils.internal.is_lazy()
            except AttributeError:
                pass

        if is_lazy:
            # Lazy mode doesn't support torch.compile
            torch._dynamo.config.disable = True
            # Enable lazy collectives for multi-HPU inference
            os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"

    def get_communicator_cls(self) -> Type:
        """
        Get the HPU communicator class.

        Returns:
            HpuCommunicator class using HCCL backend with mark_step() support.
        """
        from nanovllm.distributed import HpuCommunicator

        return HpuCommunicator

    # =========================================================================
    # Memory management methods
    # =========================================================================

    def empty_cache(self) -> None:
        """
        Clear HPU memory cache.

        Follows vllm-gaudi pattern: HPU memory management relies on
        Python garbage collection rather than torch.hpu.empty_cache().
        gc.collect() forces immediate collection of unreferenced tensors,
        allowing HPU runtime to reclaim device memory.

        See: vllm-gaudi/vllm_gaudi/v1/worker/hpu_worker.py
        """
        import gc

        gc.collect()

    def reset_peak_memory_stats(self) -> None:
        """
        Reset peak memory statistics tracking.

        HPU doesn't have native peak memory tracking like CUDA.
        This is a no-op; use HpuMemoryProfiler context manager for
        accurate delta-based memory measurement.
        """
        import gc

        gc.collect()

    def get_peak_memory(self) -> int:
        """
        Get peak allocated HPU memory in bytes.

        HPU lacks true peak tracking. Returns current memory usage,
        which after warmup represents a reasonable approximation.

        Note: In lazy mode, call synchronize() first if accurate value needed.
        See: vllm-gaudi pattern - caller responsible for sync.

        Returns:
            Current memory usage in bytes.
        """
        from nanovllm.platforms.hpu.memory import HpuMemoryProfiler

        return HpuMemoryProfiler.current_device_memory_usage()

    def get_current_memory(self) -> int:
        """
        Get currently allocated HPU memory in bytes.

        Note: In lazy mode, call synchronize() first if accurate value needed.
        See: vllm-gaudi pattern - caller responsible for sync.

        Returns:
            Current memory allocation in bytes.
        """
        from nanovllm.platforms.hpu.memory import HpuMemoryProfiler

        return HpuMemoryProfiler.current_device_memory_usage()
