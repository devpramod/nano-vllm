"""
Abstract base class for platform implementations.

Platforms abstract away device-specific operations (CUDA, HPU, etc.)
allowing nano-vllm to run on different hardware backends.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, Type

if TYPE_CHECKING:
    from nanovllm.config import Config
    from nanovllm.distributed import Communicator


class Platform(ABC):
    """
    Abstract base class for hardware platform implementations.

    Each platform (CUDA, HPU, etc.) must implement these methods to provide
    device-specific functionality for tensor operations, memory management,
    and distributed communication.

    Example:
        class CudaPlatform(Platform):
            device_name = "cuda"
            device_type = "cuda"

            def set_device(self, rank: int) -> None:
                torch.cuda.set_device(rank)
            ...
    """

    # Platform identifier (e.g., "cuda", "hpu")
    device_name: str = ""
    device_type: str = ""

    @abstractmethod
    def check_and_update_config(self, config: "Config") -> None:
        """
        Validate and update configuration for this platform.

        Args:
            config: The Config object to validate and potentially modify.

        Raises:
            ValueError: If configuration is invalid for this platform.
        """
        raise NotImplementedError

    @abstractmethod
    def get_attention_backend(self) -> Type:
        """
        Get the attention backend class for this platform.

        Returns:
            The attention backend class (e.g., FlashAttention for CUDA,
            FusedSDPA for HPU).
        """
        raise NotImplementedError

    @abstractmethod
    def set_device(self, rank: int) -> None:
        """
        Set the current device for this process.

        Args:
            rank: The device index to use.
        """
        raise NotImplementedError

    @abstractmethod
    def get_memory_info(self) -> Tuple[int, int]:
        """
        Get memory information for the current device.

        Returns:
            Tuple of (free_bytes, total_bytes).
        """
        raise NotImplementedError

    @abstractmethod
    def synchronize(self) -> None:
        """
        Synchronize the current device.

        Blocks until all operations on the device are complete.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this platform is available on the current system.

        Returns:
            True if the platform's hardware/drivers are available.
        """
        raise NotImplementedError

    def get_device_string(self) -> str:
        """
        Get the device string for tensor creation.

        Returns:
            Device string (e.g., "cuda", "hpu").
        """
        return self.device_name

    @abstractmethod
    def empty_cache(self) -> None:
        """
        Clear device memory cache.

        Frees cached memory that can be reallocated.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_peak_memory_stats(self) -> None:
        """
        Reset peak memory statistics tracking.

        After calling this, get_peak_memory() will return the peak
        since this reset, not since program start.
        """
        raise NotImplementedError

    @abstractmethod
    def get_peak_memory(self) -> int:
        """
        Get peak allocated memory in bytes.

        Returns:
            Peak memory allocation since last reset, in bytes.
            On platforms without peak tracking (HPU), returns current usage.
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_memory(self) -> int:
        """
        Get currently allocated memory in bytes.

        Returns:
            Current memory allocation in bytes.
        """
        raise NotImplementedError

    @abstractmethod
    def get_communicator_cls(self) -> Type["Communicator"]:
        """
        Get the communicator class for this platform.

        Returns:
            The communicator class (e.g., CudaCommunicator, HpuCommunicator).
        """
        raise NotImplementedError

    def create_communicator(self, world_size: int, rank: int) -> "Communicator":
        """
        Create a communicator instance for this platform.

        Args:
            world_size: Total number of processes.
            rank: Rank of current process.

        Returns:
            Communicator instance for this platform.
        """
        comm_cls = self.get_communicator_cls()
        return comm_cls(world_size, rank)

    @abstractmethod
    def get_distributed_backend(self) -> str:
        """
        Get the distributed communication backend name.

        Returns:
            Backend string for torch.distributed (e.g., 'nccl', 'hccl').
        """
        raise NotImplementedError

    @property
    def supports_cuda_graphs(self) -> bool:
        """
        Whether this platform supports CUDA graph capture.

        CUDA graphs provide optimized kernel launch for repeated operations.
        HPU uses lazy mode instead and does not support CUDA graphs.

        Returns:
            True if CUDA graphs are supported, False otherwise.
        """
        return False
