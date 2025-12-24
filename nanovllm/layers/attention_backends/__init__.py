"""
Attention backend abstractions for platform-agnostic attention operations.

This package provides:
- AttentionMetadata: Base dataclass for attention metadata
- AttentionBackend: Abstract base class for attention implementations
- CudaAttentionMetadata: CUDA-specific metadata with cu_seqlens
- CudaAttentionBackend: Flash Attention implementation
- HpuAttentionMetadata: HPU-specific metadata (from nanovllm.platforms.hpu.attention)
- HpuAttentionBackend: FusedSDPA + flat_pa implementation (from nanovllm.platforms.hpu.attention)

Note: HPU attention is in nanovllm.platforms.hpu.attention to keep HPU-specific
code co-located with the platform module.
"""
from nanovllm.layers.attention_backends.base import (
    AttentionBackend,
    AttentionMetadata,
)
from nanovllm.layers.attention_backends.cuda import (
    CudaAttentionBackend,
    CudaAttentionMetadata,
)

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "CudaAttentionBackend",
    "CudaAttentionMetadata",
]


def get_hpu_attention_classes():
    """
    Lazily import HPU attention classes.

    Returns:
        Tuple of (HpuAttentionMetadata, HpuAttentionBackend)

    Raises:
        ImportError: If HPU platform is not available.
    """
    from nanovllm.platforms.hpu.attention import (
        HpuAttentionMetadata,
        HpuAttentionBackend,
    )

    return HpuAttentionMetadata, HpuAttentionBackend
