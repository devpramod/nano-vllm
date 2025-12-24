"""
Base classes for KV cache operations.

This module provides platform-agnostic abstractions for KV cache storage,
allowing different implementations for CUDA (Triton kernel) and HPU (index_copy_).
"""
from abc import ABC, abstractmethod

import torch


class KVCacheOps(ABC):
    """
    Abstract base class for KV cache operations.

    Each platform (CUDA, HPU) implements this interface with its own
    optimized cache operations. The ops are stateless - cache tensors
    are passed as arguments.

    Example:
        ops = CudaKVCacheOps()
        ops.store(k, v, k_cache, v_cache, slot_mapping)
    """

    @abstractmethod
    def store(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """
        Store key/value tensors into the KV cache at specified slots.

        This operation scatters the input K/V tensors into the cache
        at positions specified by slot_mapping. Slots with value -1
        are skipped (used for padding tokens).

        Args:
            k: Key tensor to store. Shape: [num_tokens, num_kv_heads, head_dim]
            v: Value tensor to store. Shape: [num_tokens, num_kv_heads, head_dim]
            k_cache: Key cache tensor. Shape: [num_blocks * block_size, num_kv_heads, head_dim]
                    or [num_blocks, block_size, num_kv_heads, head_dim] depending on layout.
            v_cache: Value cache tensor. Same shape as k_cache.
            slot_mapping: Maps each token position to a cache slot.
                         Shape: [num_tokens]. Value of -1 means skip (padding).

        Note:
            This operation modifies k_cache and v_cache in-place.
        """
        raise NotImplementedError
