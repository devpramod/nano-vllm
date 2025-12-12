"""
CUDA platform implementation for NVIDIA GPUs.

This module wraps PyTorch's CUDA operations to provide a uniform
interface through the Platform abstraction.
"""

from typing import TYPE_CHECKING, Tuple, Type

import torch

from nanovllm.platforms.base import Platform

if TYPE_CHECKING:
    from nanovllm.config import Config


class CudaPlatform(Platform):
    """
    Platform implementation for NVIDIA CUDA GPUs.

    Uses PyTorch's native CUDA support along with Flash Attention
    and Triton kernels for optimized inference.
    """

    device_name = "cuda"
    device_type = "cuda"

    def check_and_update_config(self, config: "Config") -> None:
        """
        Validate and update configuration for CUDA platform.

        For CUDA, we primarily ensure the block size is compatible
        with Flash Attention requirements.

        Args:
            config: The Config object to validate and potentially modify.
        """
        # CUDA supports various block sizes, but 256 is optimal for Flash Attention
        if config.kvcache_block_size % 256 != 0:
            raise ValueError(
                f"KV cache block size must be divisible by 256 for CUDA, "
                f"got {config.kvcache_block_size}"
            )

    def get_attention_backend(self) -> Type:
        """
        Get the Flash Attention backend for CUDA.

        Returns:
            The Attention class from nanovllm.layers.attention.
        """
        from nanovllm.layers.attention import Attention

        return Attention

    def set_device(self, rank: int) -> None:
        """
        Set the current CUDA device.

        Args:
            rank: The device index (GPU number) to use.
        """
        torch.cuda.set_device(rank)

    def get_memory_info(self) -> Tuple[int, int]:
        """
        Get memory information for the current CUDA device.

        Returns:
            Tuple of (free_bytes, total_bytes).
        """
        return torch.cuda.mem_get_info()

    def synchronize(self) -> None:
        """
        Synchronize the current CUDA device.

        Blocks until all CUDA operations are complete.
        """
        torch.cuda.synchronize()

    def is_available(self) -> bool:
        """
        Check if CUDA is available on this system.

        Returns:
            True if CUDA GPUs and drivers are available.
        """
        return torch.cuda.is_available()

    def get_communicator_cls(self) -> Type:
        """
        Get the CUDA communicator class.

        Returns:
            CudaCommunicator class using NCCL backend.
        """
        from nanovllm.distributed import CudaCommunicator

        return CudaCommunicator
