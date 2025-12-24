"""
KV cache operations for platform-agnostic cache management.

This package provides:
- KVCacheOps: Abstract base class for cache operations
- CudaKVCacheOps: Triton kernel implementation for CUDA

For HPU, see nanovllm.platforms.hpu.kv_cache (Task 12).
"""
from nanovllm.layers.kv_cache.base import KVCacheOps
from nanovllm.layers.kv_cache.cuda import CudaKVCacheOps

__all__ = [
    "KVCacheOps",
    "CudaKVCacheOps",
]
