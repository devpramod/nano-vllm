"""
CUDA communicator using NCCL backend.

Thin wrapper around torch.distributed for CUDA/NCCL communication.
No additional synchronization needed - NCCL handles it internally.
"""

import torch
import torch.distributed as dist

from nanovllm.distributed.communicator import Communicator


class CudaCommunicator(Communicator):
    """
    Communicator implementation for CUDA using NCCL backend.

    This is a thin wrapper around torch.distributed calls.
    NCCL handles synchronization internally, so no mark_step() needed.
    """

    def __init__(self, world_size: int, rank: int):
        """
        Initialize CUDA communicator.

        Args:
            world_size: Total number of processes.
            rank: Rank of current process.
        """
        super().__init__(world_size, rank)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform all-reduce using NCCL.

        Args:
            tensor: Input tensor to reduce.

        Returns:
            The reduced tensor (modified in-place).
        """
        dist.all_reduce(tensor)
        return tensor

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Gather tensors from all processes and concatenate.

        Args:
            tensor: Input tensor from this process.
            dim: Dimension along which to concatenate.

        Returns:
            Concatenated tensor from all processes.
        """
        if dim < 0:
            dim += tensor.dim()

        input_size = tensor.size()

        # Allocate output tensor
        output_tensor = torch.empty(
            (self.world_size,) + input_size,
            dtype=tensor.dtype,
            device=tensor.device,
        )

        # All-gather
        dist.all_gather_into_tensor(output_tensor, tensor)

        # Reshape: move world_size dimension to target dim and flatten
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )

        return output_tensor

    def gather(
        self, tensor: torch.Tensor, dst: int = 0
    ) -> list[torch.Tensor] | None:
        """
        Gather tensors from all processes to destination rank.

        Args:
            tensor: Input tensor from this process.
            dst: Destination rank that receives all tensors.

        Returns:
            List of tensors on dst rank, None on other ranks.
        """
        if self.rank == dst:
            gather_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        else:
            gather_list = None
        dist.gather(tensor, gather_list, dst)
        return gather_list

    def barrier(self) -> None:
        """Synchronize all processes."""
        dist.barrier()
