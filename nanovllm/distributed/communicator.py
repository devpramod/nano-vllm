"""
Abstract base class for distributed communication.

Communicators abstract away platform-specific distributed operations,
enabling HPU's mark_step() requirement while keeping CUDA behavior simple.
"""

from abc import ABC, abstractmethod

import torch


class Communicator(ABC):
    """
    Abstract base class for distributed communication.

    Each platform (CUDA, HPU) implements this interface to provide
    platform-specific collective operations. HPU requires mark_step()
    before collectives in lazy mode; CUDA does not.

    Example:
        comm = HpuCommunicator(world_size=2, rank=0)
        output = comm.all_reduce(tensor)  # Includes mark_step() for HPU
    """

    def __init__(self, world_size: int, rank: int):
        """
        Initialize the communicator.

        Args:
            world_size: Total number of processes in the group.
            rank: Rank of the current process (0 to world_size-1).
        """
        self.world_size = world_size
        self.rank = rank

    @abstractmethod
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform all-reduce operation across all processes.

        Reduces the tensor across all processes so that all get the
        same final result (sum by default).

        Args:
            tensor: Input tensor to reduce.

        Returns:
            The reduced tensor (modified in-place and returned).
        """
        raise NotImplementedError

    @abstractmethod
    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Gather tensors from all processes and concatenate.

        Args:
            tensor: Input tensor from this process.
            dim: Dimension along which to concatenate gathered tensors.

        Returns:
            Concatenated tensor from all processes.
        """
        raise NotImplementedError

    @abstractmethod
    def gather(
        self, tensor: torch.Tensor, dst: int = 0
    ) -> list[torch.Tensor] | None:
        """
        Gather tensors from all processes to destination rank.

        Args:
            tensor: Input tensor from this process.
            dst: Destination rank that receives all tensors (default: 0).

        Returns:
            List of tensors from all processes on dst rank, None on other ranks.
        """
        raise NotImplementedError

    @abstractmethod
    def barrier(self) -> None:
        """
        Synchronize all processes.

        Blocks until all processes have reached this point.
        """
        raise NotImplementedError
