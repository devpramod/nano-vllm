"""
Attention layer with platform-agnostic backend support.

This module provides the Attention class that delegates to platform-specific
backends (CudaAttentionBackend, HpuAttentionBackend) for optimized execution.
"""
import torch
from torch import nn

from nanovllm.utils.context import get_context
from nanovllm.layers.attention_backends.base import AttentionBackend
from nanovllm.layers.kv_cache.base import KVCacheOps


class Attention(nn.Module):
    """
    Platform-agnostic attention layer.

    Uses platform-specific backends for attention computation and KV cache
    operations. By default, auto-detects the platform and selects the
    appropriate backend.

    Args:
        num_heads: Number of attention heads
        head_dim: Dimension per head
        scale: Softmax scale factor (typically 1/sqrt(head_dim))
        num_kv_heads: Number of key/value heads (for GQA)
        backend: Optional attention backend. If None, auto-detects.
        kv_ops: Optional KV cache ops. If None, auto-detects.

    Example:
        attn = Attention(num_heads=32, head_dim=128, scale=0.088, num_kv_heads=8)
        output = attn(q, k, v)
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        backend: AttentionBackend = None,
        kv_ops: KVCacheOps = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

        # Auto-detect platform if backend/kv_ops not provided
        if backend is None or kv_ops is None:
            from nanovllm.platforms import get_platform
            platform = get_platform()
            self.backend = backend if backend else platform.get_attention_backend()
            self.kv_ops = kv_ops if kv_ops else platform.get_kv_cache_ops()
        else:
            self.backend = backend
            self.kv_ops = kv_ops

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute attention with KV cache support.

        Uses the global Context (from get_context()) for metadata like
        cu_seqlens, slot_mapping, etc. This preserves backward compatibility
        with existing code that sets context via set_context().

        Args:
            q: Query tensor [num_tokens, num_heads, head_dim]
            k: Key tensor [num_tokens, num_kv_heads, head_dim]
            v: Value tensor [num_tokens, num_kv_heads, head_dim]

        Returns:
            Attention output [num_tokens, num_heads, head_dim]
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Store K/V to cache if cache is allocated
        if k_cache.numel() and v_cache.numel():
            self.kv_ops.store(k, v, k_cache, v_cache, context.slot_mapping)

        # Build platform-specific metadata from global context
        # Each backend creates its own metadata type with the fields it needs
        metadata = self.backend.create_metadata(context)

        # Delegate to backend
        output = self.backend.forward(
            q, k, v,
            k_cache, v_cache,
            metadata,
            self.scale,
        )

        return output
