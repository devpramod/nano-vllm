"""
CUDA KV cache operations using Triton kernel.

This module provides optimized KV cache storage for CUDA using a custom
Triton kernel that efficiently scatters K/V tensors into the cache.
"""
import torch
import triton
import triton.language as tl

from nanovllm.layers.kv_cache.base import KVCacheOps


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    Triton kernel for storing K/V into cache.

    Each program instance handles one token position, copying its K/V
    to the cache slot specified by slot_mapping.

    Args:
        key_ptr: Pointer to key tensor (flattened)
        key_stride: Stride between tokens in key tensor
        value_ptr: Pointer to value tensor (flattened)
        value_stride: Stride between tokens in value tensor
        k_cache_ptr: Pointer to key cache (flattened)
        v_cache_ptr: Pointer to value cache (flattened)
        slot_mapping_ptr: Pointer to slot mapping tensor
        D: Dimension per token (num_kv_heads * head_dim)
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)

    # Skip padding tokens (slot == -1)
    if slot == -1:
        return

    # Compute offsets for input tensors
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    # Load key and value for this token
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # Compute cache offsets and store
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


class CudaKVCacheOps(KVCacheOps):
    """
    CUDA KV cache operations using Triton kernel.

    Uses a custom Triton kernel for efficient scatter of K/V tensors
    into the paged KV cache. Each token's K/V is stored at the slot
    position specified by slot_mapping.
    """

    def store(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """
        Store K/V into cache using Triton kernel.

        Args:
            k: Key tensor [num_tokens, num_kv_heads, head_dim]
            v: Value tensor [num_tokens, num_kv_heads, head_dim]
            k_cache: Key cache [num_slots, num_kv_heads * head_dim] (flattened)
            v_cache: Value cache (same shape as k_cache)
            slot_mapping: Slot indices [num_tokens], -1 for padding
        """
        N, num_heads, head_dim = k.shape
        D = num_heads * head_dim

        # Validate tensor layouts for kernel assumptions
        assert k.stride(-1) == 1 and v.stride(-1) == 1, "K/V must be contiguous in last dim"
        assert k.stride(1) == head_dim and v.stride(1) == head_dim, "K/V head dim stride mismatch"
        assert k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1, "Cache must be contiguous"
        assert slot_mapping.numel() == N, "slot_mapping length must match num_tokens"

        # Launch kernel with one program per token
        store_kvcache_kernel[(N,)](
            k, k.stride(0),
            v, v.stride(0),
            k_cache, v_cache,
            slot_mapping,
            D,
        )
