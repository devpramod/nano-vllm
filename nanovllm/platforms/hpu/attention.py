"""
HPU attention backend using Habana's FusedSDPA and flat_pa.

This module provides the HPU-specific implementation of attention operations:
- Prefill: Uses FusedSDPA for batched prompt attention
- Decode: Uses flat_pa for block-based paged attention

Reference: vllm-gaudi/vllm_gaudi/extension/ops.py for flat_pa implementation
"""
from dataclasses import dataclass, field
from typing import Optional

import torch

from nanovllm.layers.attention_backends.base import AttentionBackend, AttentionMetadata


@dataclass
class HpuAttentionMetadata(AttentionMetadata):
    """
    HPU-specific attention metadata for FusedSDPA and flat_pa.

    Extends base AttentionMetadata with fields required by:
    - FusedSDPA (prefill): attn_bias, seq_lens_tensor
    - flat_pa (decode): block_list, block_mapping, block_groups, block_bias

    Attributes:
        is_prefill: Inherited. True for prefill phase, False for decode.
        slot_mapping: Inherited. Maps tokens to KV cache slots.
        block_tables: Inherited. Block table for paged attention.

        # Prefill-specific (FusedSDPA)
        attn_bias: Attention bias tensor for causal masking in prefill.
                   Shape: [batch, 1, max_seq_len, max_seq_len] or None if using is_causal=True.
        seq_lens_tensor: Tensor of sequence lengths for variable-length batches.
                        Shape: [batch_size]. Used for valid_sequence_lengths in FusedSDPA.

        # Decode-specific (flat_pa)
        block_list: Flattened list of block indices to fetch from KV cache.
                   Shape: [total_blocks_needed]. Used in index_select for K/V fetch.
        block_mapping: Sparse matrix for batch2block/block2batch scatter-gather.
                      Shape: [num_blocks, batch_size]. Each row is one-hot indicating
                      which batch item the block belongs to.
        block_groups: Maps each block to its sequence index.
                     Shape: [num_blocks]. Used for cross-block softmax normalization.
        block_bias: Attention bias per block for masking invalid positions.
                   Shape: [num_blocks, 1, 1, block_size]. Masks padding within blocks.
        context_lens_tensor: Total context length per sequence (for decode).
                            Shape: [batch_size]. Number of tokens in KV cache.

        # Configuration
        block_size: KV cache block size. Must be 128 for HPU.
    """

    # =========================================================================
    # Prefill-specific fields (FusedSDPA)
    # =========================================================================
    attn_bias: Optional[torch.Tensor] = None
    seq_lens_tensor: Optional[torch.Tensor] = None

    # =========================================================================
    # Decode-specific fields (flat_pa)
    # =========================================================================
    block_list: Optional[torch.Tensor] = None
    block_mapping: Optional[torch.Tensor] = None
    block_groups: Optional[torch.Tensor] = None
    block_bias: Optional[torch.Tensor] = None
    context_lens_tensor: Optional[torch.Tensor] = None

    # =========================================================================
    # Configuration
    # =========================================================================
    block_size: int = 128


# =============================================================================
# flat_pa helper functions (ported from vllm-gaudi/vllm_gaudi/extension/ops.py)
# =============================================================================


def batch2block(
    tensor: torch.Tensor,
    block_mapping: torch.Tensor,
    matmul_op=torch.matmul,
) -> torch.Tensor:
    """
    Scatter tensor from batch dimension to block dimension.

    Uses matrix multiplication for efficient scatter operation.
    block_mapping is a sparse matrix where each column is one-hot,
    indicating which batch item each block belongs to.

    Args:
        tensor: Input tensor [batch_size, ...].
        block_mapping: Scatter matrix [num_blocks, batch_size].
        matmul_op: Matrix multiply function (allows custom ops).

    Returns:
        Scattered tensor [num_blocks, ...].

    Reference: vllm_gaudi/extension/ops.py lines 49-50
    """
    shape = tuple(tensor.shape)
    return matmul_op(block_mapping, tensor.view(shape[0], -1)).view(-1, *shape[1:])


def block2batch(
    tensor: torch.Tensor,
    block_mapping: torch.Tensor,
    matmul_op=torch.matmul,
) -> torch.Tensor:
    """
    Gather tensor from block dimension back to batch dimension.

    Inverse of batch2block - uses transposed block_mapping.

    Args:
        tensor: Input tensor [num_blocks, ...].
        block_mapping: Scatter matrix [num_blocks, batch_size].
        matmul_op: Matrix multiply function (allows custom ops).

    Returns:
        Gathered tensor [batch_size, ...].

    Reference: vllm_gaudi/extension/ops.py lines 53-54
    """
    shape = tuple(tensor.shape)
    return matmul_op(block_mapping.t(), tensor.view(shape[0], -1)).view(-1, *shape[1:])


def grouped_max(
    block_max: torch.Tensor,
    batch_size: int,
    block_groups: torch.Tensor,
) -> torch.Tensor:
    """
    Compute maximum across blocks belonging to the same sequence.

    Used for numerically stable softmax across paged attention blocks.
    Each sequence may have multiple blocks; this finds the max within
    each sequence's blocks.

    Args:
        block_max: Per-block maximum values [num_blocks, ...].
        batch_size: Number of sequences in batch.
        block_groups: Maps blocks to sequence indices [num_blocks].

    Returns:
        Grouped maximum broadcast back to blocks [num_blocks, ...].

    Reference: vllm_gaudi/extension/ops.py lines 34-41
    """
    import math

    # Create output with -inf for aggregation
    group_max = torch.full(
        [batch_size + 1, *block_max.shape[1:]],
        -math.inf,
        dtype=block_max.dtype,
        device=block_max.device,
    )
    # Aggregate max per sequence using index_reduce
    group_max = group_max.index_reduce_(0, block_groups, block_max, "amax")
    # Broadcast back to block dimension
    group_max = group_max.index_select(0, block_groups)
    return group_max


def pipelined_pa(
    attn: torch.Tensor,
    value: torch.Tensor,
    block_bias: Optional[torch.Tensor],
    block_groups: torch.Tensor,
    block_mapping: torch.Tensor,
    batch_size: int,
    matmul_av_op=torch.matmul,
    batch2block_matmul_op=torch.matmul,
    block2batch_matmul_op=torch.matmul,
) -> torch.Tensor:
    """
    Block-level paged attention with numerically stable softmax.

    This is the core of flat_pa: computes attention across blocks while
    maintaining numerical stability through cross-block max/sum normalization.

    Algorithm:
    1. Apply block_bias (mask invalid positions)
    2. Compute per-block max for stability
    3. Subtract max and compute exp (block-local softmax)
    4. Compute per-block sum
    5. Multiply attn weights with values
    6. Rescale using cross-block max/sum for global normalization

    Args:
        attn: Attention scores [num_blocks, q_heads, 1, block_size].
        value: Value tensor [num_blocks, kv_heads, block_size, head_dim].
        block_bias: Attention mask [num_blocks, 1, 1, block_size] or None.
        block_groups: Block to sequence mapping [num_blocks].
        block_mapping: Scatter matrix [num_blocks, batch_size].
        batch_size: Number of sequences.
        matmul_av_op: Matmul op for attn @ value.
        batch2block_matmul_op: Matmul op for batch2block.
        block2batch_matmul_op: Matmul op for block2batch.

    Returns:
        Weighted values [num_blocks, q_heads, 1, head_dim].

    Reference: vllm_gaudi/extension/ops.py lines 65-114
    """
    # Match dtypes for block_bias
    if block_bias is not None and attn.dtype != block_bias.dtype:
        block_bias = block_bias.to(dtype=attn.dtype)

    # Apply attention mask
    if block_bias is not None:
        attn = attn + block_bias

    # Block-local softmax computation
    block_max = attn.amax(dim=-1, keepdim=True)
    attn = attn - block_max
    attn = attn.exp()

    # Convert back to value dtype if needed (e.g., from fp32 softmax)
    if attn.dtype != value.dtype:
        attn = attn.to(value.dtype)

    block_sums = attn.sum(dim=-1, keepdim=True)

    # Compute attention @ value
    attn = matmul_av_op(attn, value)

    # Cross-block normalization for numerical stability
    adjustment_target_shape = block_max.shape
    block_max = block_max.squeeze((-1, -2))
    block_sums = block_sums.squeeze((-1, -2))

    # Get maximum across all blocks of the same sequence
    group_max = grouped_max(block_max, batch_size, block_groups)

    # Compute adjustment factor: exp(block_max - group_max)
    block_adjustment = (block_max - group_max).exp()
    if block_adjustment.dtype != value.dtype:
        block_adjustment = block_adjustment.to(value.dtype)

    # Adjust sums by the max adjustment
    sum_adjusted = block_sums * block_adjustment

    # Aggregate adjusted sums per sequence and broadcast back
    group_sum_adjusted = block2batch(sum_adjusted, block_mapping, block2batch_matmul_op)
    group_sum_adjusted = batch2block(group_sum_adjusted, block_mapping, batch2block_matmul_op)

    # Reshape for broadcasting
    sum_adjusted = sum_adjusted.view(*adjustment_target_shape)
    group_sum_adjusted = group_sum_adjusted.view(*adjustment_target_shape)
    block_adjustment = block_adjustment.view(*adjustment_target_shape)

    # Stability: ensure we don't divide by zero
    group_sum_adjusted = torch.maximum(group_sum_adjusted, sum_adjusted)

    # Final rescaling
    rescale = block_adjustment / group_sum_adjusted
    attn = attn * rescale

    return attn


# =============================================================================
# FusedSDPA kernel loading and wrapper
# =============================================================================


def _kernel_loader(name: str):
    """
    Decorator factory for lazy kernel loading with graceful fallback.

    Creates a cached loader that attempts to import the kernel once
    and returns None if import fails.

    Args:
        name: Kernel name for logging.

    Returns:
        Decorator that wraps a kernel import function.

    Reference: vllm_gaudi/extension/kernels.py lines 11-26
    """
    from functools import cache

    def loader(fn):
        @cache
        def loader_impl():
            try:
                return fn()
            except (ImportError, AttributeError):
                import logging

                logging.warning(
                    f"Could not import HPU {name} kernel. "
                    "Will use PyTorch fallback."
                )
                return None

        return loader_impl

    return loader


@_kernel_loader("FusedSDPA")
def _get_fsdpa_kernel():
    """
    Load Habana's FusedSDPA kernel.

    Returns:
        FusedSDPA kernel class, or None if unavailable.
    """
    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    return FusedSDPA


class ModuleFusedSDPA(torch.nn.Module):
    """
    nn.Module wrapper for Habana's FusedSDPA kernel.

    Provides a clean interface with named arguments and handles
    the variable argument signature for window attention.

    This wrapper is useful for:
    - Clean argument marshaling (FusedSDPA.apply uses positional args)
    - HPU Graph compatibility (nn.Module works with wrap_in_hpu_graph)
    - Easy mocking in tests

    Reference: vllm_gaudi/extension/utils.py lines 131-159
    """

    def __init__(self, fused_sdpa=None):
        """
        Initialize the wrapper.

        Args:
            fused_sdpa: The FusedSDPA kernel. If None, attempts to load.

        Raises:
            RuntimeError: If kernel is None and cannot be loaded.
        """
        super().__init__()
        if fused_sdpa is None:
            fused_sdpa = _get_fsdpa_kernel()
        if fused_sdpa is None:
            raise RuntimeError(
                "FusedSDPA kernel not available. "
                "Ensure habana_frameworks is installed."
            )
        self._fsdpa = fused_sdpa

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = True,
        scale: Optional[float] = None,
        softmax_mode: str = "fast",
        recompute_mode: bool = True,
        valid_sequence_lengths: Optional[torch.Tensor] = None,
        padding_side: str = "left",
        window_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Execute FusedSDPA attention.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim].
            key: Key tensor [batch, seq_len, num_kv_heads, head_dim].
            value: Value tensor [batch, seq_len, num_kv_heads, head_dim].
            attn_mask: Optional attention mask. None for causal-only.
            dropout_p: Dropout probability (0.0 for inference).
            is_causal: Whether to apply causal mask.
            scale: Softmax scale. None uses 1/sqrt(head_dim).
            softmax_mode: 'fast' or 'accurate'. Use 'fast' for inference.
            recompute_mode: If True, recomputes attention for memory efficiency.
            valid_sequence_lengths: Per-sequence valid lengths for variable-length.
            padding_side: 'left' or 'right' padding.
            window_size: Optional sliding window size for local attention.

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim].
        """
        if window_size is not None:
            return self._fsdpa.apply(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                softmax_mode,
                recompute_mode,
                valid_sequence_lengths,
                padding_side,
                False,  # attn_mask_type
                False,  # attn_mask_format
                window_size,
            )
        else:
            return self._fsdpa.apply(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                softmax_mode,
                recompute_mode,
                valid_sequence_lengths,
                padding_side,
            )


# =============================================================================
# HpuAttentionBackend implementation
# =============================================================================


class HpuAttentionBackend(AttentionBackend):
    """
    HPU attention backend using FusedSDPA for prefill and flat_pa for decode.

    Architecture:
    - Prefill: Uses Habana's FusedSDPA kernel for batched prompt attention.
              Supports variable-length sequences via seq_lens_tensor.
    - Decode: Uses flat_pa algorithm for block-based paged attention.
              Numerically stable softmax across scattered KV cache blocks.

    Fallback:
    - If FusedSDPA unavailable, uses F.scaled_dot_product_attention.
    - Fallback is triggered on first failure and cached.
    """

    def __init__(self):
        """Initialize the HPU attention backend."""
        self._fsdpa: Optional[ModuleFusedSDPA] = None
        self._use_fallback = False
        self._htorch = None

        # Try to load Habana frameworks
        try:
            import habana_frameworks.torch as htorch

            self._htorch = htorch
        except ImportError:
            pass

    def _get_fsdpa(self) -> Optional[ModuleFusedSDPA]:
        """Lazily initialize FusedSDPA wrapper."""
        if self._fsdpa is None and not self._use_fallback:
            try:
                self._fsdpa = ModuleFusedSDPA()
            except RuntimeError:
                self._use_fallback = True
        return self._fsdpa

    def _mark_step(self) -> None:
        """Insert mark_step() for lazy mode synchronization."""
        if self._htorch is not None:
            self._htorch.core.mark_step()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: HpuAttentionMetadata,
        scale: float,
    ) -> torch.Tensor:
        """
        Execute attention using FusedSDPA (prefill) or flat_pa (decode).

        Args:
            q: Query tensor [num_tokens, num_heads, head_dim]
            k: Key tensor [num_tokens, num_kv_heads, head_dim]
            v: Value tensor [num_tokens, num_kv_heads, head_dim]
            k_cache: Key cache [num_blocks * block_size, num_kv_heads, head_dim]
            v_cache: Value cache (same shape as k_cache)
            metadata: HpuAttentionMetadata with prefill/decode-specific fields
            scale: Softmax scale factor (1/sqrt(head_dim))

        Returns:
            Attention output [num_tokens, num_heads, head_dim]
        """
        try:
            if metadata.is_prefill:
                output = self._forward_prefill(q, k, v, metadata, scale)
            else:
                output = self._forward_decode(q, k_cache, v_cache, metadata, scale)
        except Exception as e:
            if not self._use_fallback:
                import logging

                logging.warning(
                    f"FusedSDPA/flat_pa failed: {e}. Using PyTorch fallback."
                )
                self._use_fallback = True
            output = self._forward_fallback(q, k, v, metadata, scale)

        # Mark step for lazy mode - triggers execution of accumulated ops
        self._mark_step()

        return output

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: HpuAttentionMetadata,
        scale: float,
    ) -> torch.Tensor:
        """
        Prefill attention using FusedSDPA.

        Processes multiple tokens per sequence (prompt processing).
        Uses FusedSDPA with causal masking and variable-length support.

        Input shapes:
            q: [num_tokens, num_heads, head_dim]
            k: [num_tokens, num_kv_heads, head_dim]
            v: [num_tokens, num_kv_heads, head_dim]

        FusedSDPA expects:
            [batch, seq_len, num_heads, head_dim]

        For simplicity, we treat the entire batch as one sequence
        and use attn_bias for proper causal masking per sub-sequence.
        """
        num_tokens, num_heads, head_dim = q.shape
        num_kv_heads = k.shape[1]

        fsdpa = self._get_fsdpa()
        if fsdpa is None:
            return self._forward_fallback(q, k, v, metadata, scale)

        # Reshape to FusedSDPA format: [batch=1, seq_len, num_heads, head_dim]
        q_fsdpa = q.unsqueeze(0)  # [1, num_tokens, num_heads, head_dim]
        k_fsdpa = k.unsqueeze(0)  # [1, num_tokens, num_kv_heads, head_dim]
        v_fsdpa = v.unsqueeze(0)  # [1, num_tokens, num_kv_heads, head_dim]

        # Handle GQA: expand K/V heads to match Q heads
        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            k_fsdpa = k_fsdpa.repeat_interleave(repeat_factor, dim=2)
            v_fsdpa = v_fsdpa.repeat_interleave(repeat_factor, dim=2)

        # Call FusedSDPA
        output = fsdpa(
            query=q_fsdpa,
            key=k_fsdpa,
            value=v_fsdpa,
            attn_mask=metadata.attn_bias,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
            softmax_mode="fast",
            recompute_mode=True,
            valid_sequence_lengths=metadata.seq_lens_tensor,
            padding_side="left",
        )

        # Reshape back: [1, num_tokens, num_heads, head_dim] -> [num_tokens, num_heads, head_dim]
        return output.squeeze(0)

    def _forward_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: HpuAttentionMetadata,
        scale: float,
    ) -> torch.Tensor:
        """
        Decode attention using flat_pa algorithm.

        Processes single token per sequence (autoregressive generation).
        Uses block-based paged attention with cross-block softmax normalization.

        Input shapes:
            q: [batch_size, num_heads, head_dim] (one token per sequence)
            k_cache: [num_blocks * block_size, num_kv_heads, head_dim]
            v_cache: [num_blocks * block_size, num_kv_heads, head_dim]

        Algorithm:
        1. Scatter query to block dimension using block_mapping
        2. Fetch K/V from cache using block_list
        3. Compute Q @ K^T attention scores
        4. Apply pipelined_pa for stable cross-block softmax
        5. Gather results back to batch dimension
        """
        batch_size, num_heads, head_dim = q.shape
        num_kv_heads = k_cache.shape[1]
        block_size = metadata.block_size

        # Reshape query for flat_pa: [batch, num_heads * head_dim]
        q_flat = q.view(batch_size, -1)

        # Scatter query to blocks: [num_blocks, num_heads * head_dim]
        q_blocks = batch2block(scale * q_flat, metadata.block_mapping)

        # Reshape for attention: [num_blocks, num_heads, 1, head_dim]
        q_blocks = q_blocks.view(-1, num_heads, 1, head_dim)

        # Fetch K/V from cache using block_list
        # Cache shape: [num_blocks * block_size, num_kv_heads, head_dim]
        # Reshape to: [num_blocks, block_size, num_kv_heads, head_dim]
        k_blocked = k_cache.view(-1, block_size, num_kv_heads, head_dim)
        v_blocked = v_cache.view(-1, block_size, num_kv_heads, head_dim)

        # Select blocks: [num_selected_blocks, block_size, num_kv_heads, head_dim]
        k_selected = k_blocked.index_select(0, metadata.block_list)
        v_selected = v_blocked.index_select(0, metadata.block_list)

        # Transpose for attention: [num_blocks, num_kv_heads, block_size, head_dim]
        k_selected = k_selected.transpose(1, 2)
        v_selected = v_selected.transpose(1, 2)

        # Handle GQA: unflatten heads for grouped attention
        if num_kv_heads != num_heads:
            # q: [num_blocks, num_heads, 1, head_dim]
            # -> [num_blocks, num_kv_heads, heads_per_kv, 1, head_dim]
            heads_per_kv = num_heads // num_kv_heads
            q_blocks = q_blocks.unflatten(1, (num_kv_heads, heads_per_kv))
            # k: [num_blocks, num_kv_heads, block_size, head_dim]
            # -> [num_blocks, num_kv_heads, 1, block_size, head_dim]
            k_selected = k_selected.unsqueeze(2)
            v_selected = v_selected.unsqueeze(2)

        # Compute attention scores: Q @ K^T
        # k_selected transposed: [num_blocks, num_kv_heads, (1,) head_dim, block_size]
        k_t = k_selected.transpose(-2, -1)
        attn = torch.matmul(q_blocks, k_t)

        # Reshape block_bias for broadcasting
        block_bias = metadata.block_bias
        if block_bias is not None:
            block_bias = block_bias.view(k_selected.size(0), 1, 1, -1)
            if num_kv_heads != num_heads:
                block_bias = block_bias.unsqueeze(2)

        # Apply pipelined_pa for numerically stable softmax
        attn = pipelined_pa(
            attn=attn,
            value=v_selected,
            block_bias=block_bias,
            block_groups=metadata.block_groups,
            block_mapping=metadata.block_mapping,
            batch_size=batch_size,
        )

        # Gather back to batch dimension
        attn = block2batch(attn.view(-1, num_heads * head_dim), metadata.block_mapping)

        # Squeeze and reshape: [batch, num_heads, head_dim]
        attn = attn.view(batch_size, num_heads, head_dim)

        return attn

    def _forward_fallback(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: HpuAttentionMetadata,
        scale: float,
    ) -> torch.Tensor:
        """
        Fallback attention using PyTorch's scaled_dot_product_attention.

        Used when FusedSDPA is unavailable or fails.
        """
        import torch.nn.functional as F

        num_tokens, num_heads, head_dim = q.shape
        num_kv_heads = k.shape[1]

        # Handle GQA
        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # Reshape for SDPA: [batch=1, num_heads, seq_len, head_dim]
        q = q.transpose(0, 1).unsqueeze(0)  # [1, num_heads, num_tokens, head_dim]
        k = k.transpose(0, 1).unsqueeze(0)  # [1, num_heads, num_tokens, head_dim]
        v = v.transpose(0, 1).unsqueeze(0)  # [1, num_heads, num_tokens, head_dim]

        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=metadata.attn_bias, scale=scale, is_causal=True
        )

        # Reshape back: [1, num_heads, num_tokens, head_dim] -> [num_tokens, num_heads, head_dim]
        return output.squeeze(0).transpose(0, 1)

    def create_metadata(self, context) -> HpuAttentionMetadata:
        """
        Create HPU attention metadata from execution context.

        For prefill: Uses seq_lens and attn_bias for FusedSDPA.
        For decode: Uses block_list, block_mapping, block_groups for flat_pa.

        The context is expected to have HPU-specific fields set by the
        model runner's prepare_prefill/prepare_decode methods.

        Args:
            context: Execution context with HPU-specific fields.

        Returns:
            HpuAttentionMetadata for FusedSDPA (prefill) or flat_pa (decode).
        """
        # Base fields
        metadata = HpuAttentionMetadata(
            is_prefill=context.is_prefill,
            slot_mapping=context.slot_mapping,
            block_tables=context.block_tables,
        )

        # Prefill-specific fields
        if context.is_prefill:
            metadata.attn_bias = getattr(context, "attn_bias", None)
            metadata.seq_lens_tensor = getattr(context, "seq_lens_tensor", None)
        else:
            # Decode-specific fields for flat_pa
            metadata.block_list = getattr(context, "block_list", None)
            metadata.block_mapping = getattr(context, "block_mapping", None)
            metadata.block_groups = getattr(context, "block_groups", None)
            metadata.block_bias = getattr(context, "block_bias", None)
            metadata.context_lens_tensor = getattr(context, "context_lens_tensor", None)

        # Configuration
        metadata.block_size = getattr(context, "block_size", 128)

        return metadata
