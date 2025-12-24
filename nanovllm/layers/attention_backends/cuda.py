"""
CUDA attention backend using Flash Attention.

This module provides the CUDA-specific implementation of attention operations
using the flash_attn library for both prefill and decode phases.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanovllm.layers.attention_backends.base import AttentionBackend, AttentionMetadata


@dataclass
class CudaAttentionMetadata(AttentionMetadata):
    """
    CUDA-specific attention metadata for Flash Attention.

    Extends base AttentionMetadata with fields required by flash_attn_varlen_func
    and flash_attn_with_kvcache.

    Attributes:
        cu_seqlens_q: Cumulative sequence lengths for queries. Shape: [batch_size + 1].
                     Example: [0, 128, 256] means 2 sequences of 128 tokens each.
        cu_seqlens_k: Cumulative sequence lengths for keys. Shape: [batch_size + 1].
        max_seqlen_q: Maximum query sequence length in the batch.
        max_seqlen_k: Maximum key sequence length in the batch.
        context_lens: Sequence lengths for decode phase. Shape: [batch_size].
                     Represents how many tokens are already in KV cache per sequence.
    """
    cu_seqlens_q: torch.Tensor = None
    cu_seqlens_k: torch.Tensor = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    context_lens: Optional[torch.Tensor] = None


class CudaAttentionBackend(AttentionBackend):
    """
    CUDA attention backend using Flash Attention.

    Uses flash_attn_varlen_func for prefill (variable-length batched attention)
    and flash_attn_with_kvcache for decode (single-token attention with KV cache).
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: CudaAttentionMetadata,
        scale: float,
    ) -> torch.Tensor:
        """
        Execute attention using Flash Attention kernels.

        For prefill: Uses flash_attn_varlen_func with cu_seqlens for variable-length
                    batched attention. Supports prefix caching via block_tables.

        For decode: Uses flash_attn_with_kvcache for efficient single-token attention
                   against the KV cache.

        Args:
            q: Query tensor [num_tokens, num_heads, head_dim]
            k: Key tensor [num_tokens, num_kv_heads, head_dim]
            v: Value tensor [num_tokens, num_kv_heads, head_dim]
            k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim]
            v_cache: Value cache (same shape as k_cache)
            metadata: CudaAttentionMetadata with cu_seqlens, etc.
            scale: Softmax scale factor

        Returns:
            Attention output [num_tokens, num_heads, head_dim]
        """
        if metadata.is_prefill:
            # Prefill: process multiple tokens per sequence
            if metadata.block_tables is not None:
                # Prefix caching: use cached K/V from block tables
                k_input, v_input = k_cache, v_cache
            else:
                # No prefix caching: use input K/V directly
                k_input, v_input = k, v

            output = flash_attn_varlen_func(
                q, k_input, v_input,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k=metadata.cu_seqlens_k,
                max_seqlen_q=metadata.max_seqlen_q,
                max_seqlen_k=metadata.max_seqlen_k,
                softmax_scale=scale,
                causal=True,
                block_table=metadata.block_tables,
            )
        else:
            # Decode: process one token per sequence
            # flash_attn_with_kvcache expects q shape: [batch, 1, num_heads, head_dim]
            output = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=metadata.context_lens,
                block_table=metadata.block_tables,
                softmax_scale=scale,
                causal=True,
            )
            # Squeeze back to [batch, num_heads, head_dim]
            output = output.squeeze(1)

        return output

    def create_metadata(self, context) -> CudaAttentionMetadata:
        """
        Create CUDA attention metadata from execution context.

        Args:
            context: Execution context with CUDA-specific fields.

        Returns:
            CudaAttentionMetadata with cu_seqlens and related fields.
        """
        return CudaAttentionMetadata(
            is_prefill=context.is_prefill,
            slot_mapping=context.slot_mapping,
            block_tables=context.block_tables,
            cu_seqlens_q=context.cu_seqlens_q,
            cu_seqlens_k=context.cu_seqlens_k,
            max_seqlen_q=context.max_seqlen_q,
            max_seqlen_k=context.max_seqlen_k,
            context_lens=context.context_lens,
        )
