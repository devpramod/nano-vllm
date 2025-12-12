"""
HPU communicator using HCCL backend.

Key difference from CUDA: must call htorch.core.mark_step() BEFORE
distributed collectives in lazy mode. This triggers execution of
accumulated lazy operations before communication.
"""

import torch
import torch.distributed as dist

from nanovllm.distributed.communicator import Communicator

# Import Habana frameworks with graceful fallback
try:
    import habana_frameworks.torch as htorch

    _HABANA_AVAILABLE = True
except ImportError:
    htorch = None
    _HABANA_AVAILABLE = False


class HpuCommunicator(Communicator):
    """
    Communicator implementation for HPU using HCCL backend.

    Critical: mark_step() is called BEFORE each collective operation.
    This is required for lazy mode execution - it triggers the execution
    of accumulated operations before communication can proceed.

    See vllm-gaudi/vllm_gaudi/distributed/device_communicators/hpu_communicator.py
    """

    def __init__(self, world_size: int, rank: int):
        """
        Initialize HPU communicator.

        Args:
            world_size: Total number of processes.
            rank: Rank of current process.
        """
        super().__init__(world_size, rank)

    def _mark_step(self) -> None:
        """
        Insert lazy mode synchronization point.

        Must be called before distributed collectives to ensure
        all pending operations are executed first.
        """
        if htorch is not None:
            htorch.core.mark_step()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform all-reduce using HCCL with lazy mode sync.

        Args:
            tensor: Input tensor to reduce.

        Returns:
            The reduced tensor (modified in-place).
        """
        # Critical: mark_step BEFORE collective
        self._mark_step()
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

        # Critical: mark_step BEFORE collective
        self._mark_step()
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
        # Critical: mark_step BEFORE collective
        self._mark_step()
        dist.gather(tensor, gather_list, dst)
        return gather_list

    def barrier(self) -> None:
        """Synchronize all processes with lazy mode sync."""
        # Critical: mark_step BEFORE barrier to flush pending ops
        self._mark_step()
        dist.barrier()
