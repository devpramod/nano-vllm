"""
Base classes for attention backends.

This module provides platform-agnostic abstractions for attention operations,
allowing different implementations for CUDA (Flash Attention) and HPU (FusedSDPA).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class AttentionMetadata:
    """
    Base metadata for attention operations.

    Contains fields common to all platforms. Platform-specific subclasses
    add their own fields (e.g., cu_seqlens for CUDA, attn_bias for HPU).

    Attributes:
        is_prefill: True for prefill phase (processing prompt), False for decode
        slot_mapping: Maps token positions to KV cache slots. Shape: [num_tokens].
                     Value of -1 indicates padding (no cache write).
        block_tables: Block table for paged attention. Shape: [batch_size, max_blocks].
                     None during prefill without prefix caching.
    """
    is_prefill: bool
    slot_mapping: torch.Tensor
    block_tables: Optional[torch.Tensor] = None


class AttentionBackend(ABC):
    """
    Abstract base class for attention backends.

    Each platform (CUDA, HPU) implements this interface with its own
    optimized attention kernel. The backend is stateless - all state
    is passed via the metadata parameter.

    Example:
        backend = CudaAttentionBackend()
        output = backend.forward(q, k, v, k_cache, v_cache, metadata, scale)
    """

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: AttentionMetadata,
        scale: float,
    ) -> torch.Tensor:
        """
        Unified attention forward for both prefill and decode.

        The backend checks metadata.is_prefill to determine which code path
        to take internally. This unified interface allows the Attention class
        to be platform-agnostic.

        Args:
            q: Query tensor. Shape: [num_tokens, num_heads, head_dim]
            k: Key tensor. Shape: [num_tokens, num_kv_heads, head_dim]
            v: Value tensor. Shape: [num_tokens, num_kv_heads, head_dim]
            k_cache: Key cache. Shape: [num_blocks, block_size, num_kv_heads, head_dim]
            v_cache: Value cache. Same shape as k_cache.
            metadata: Platform-specific attention metadata (subclass of AttentionMetadata)
            scale: Softmax scale factor (typically 1/sqrt(head_dim))

        Returns:
            Attention output. Shape: [num_tokens, num_heads, head_dim]
        """
        raise NotImplementedError

    @abstractmethod
    def create_metadata(self, context) -> AttentionMetadata:
        """
        Create platform-specific attention metadata from execution context.

        This factory method allows each backend to construct its own
        metadata type with the fields it needs.

        Args:
            context: Execution context with fields like is_prefill, slot_mapping,
                    block_tables, cu_seqlens_*, etc.

        Returns:
            Platform-specific AttentionMetadata subclass instance.
        """
        raise NotImplementedError
