"""
Linear bucketing for HPU graph warmup.

Bucketing reduces the number of unique HPU graph compilations by padding
inputs to predefined bucket sizes. During warmup, we pre-compile graphs
for all bucket shapes so inference never triggers runtime compilation.

Example:
    Without bucketing:
        Request (bs=3, seq=412) -> Compiles new graph (slow!)
        Request (bs=3, seq=413) -> Compiles new graph (slow!)

    With bucketing (buckets at seq=[128, 256, 512]):
        Request (bs=3, seq=412) -> Pads to (4, 512) -> Reuses pre-compiled graph
        Request (bs=3, seq=413) -> Pads to (4, 512) -> Reuses pre-compiled graph
"""

import bisect
import logging
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def warmup_range(min_val: int, step: int, max_val: int) -> List[int]:
    """
    Generate warmup range with ramp-up phase.

    The ramp-up phase doubles values from min until reaching step,
    then uses linear increments of step until max.

    This creates dense buckets for small values (common case) and
    sparser buckets for large values (rare case).

    Example:
        min=2, step=32, max=64
        => ramp_up = [2, 4, 8, 16]
        => stable = [32, 64]
        => result = [2, 4, 8, 16, 32, 64]

    Args:
        min_val: Minimum bucket value
        step: Step size for linear phase (also threshold for ramp-up)
        max_val: Maximum bucket value

    Returns:
        Sorted list of bucket values
    """
    if min_val > max_val:
        raise ValueError(f"min ({min_val}) cannot be greater than max ({max_val})")

    buckets = []

    # Ramp-up phase: double until we reach step
    val = min_val
    while val < step and val <= max_val:
        buckets.append(val)
        val *= 2

    # Stable phase: linear increments of step
    for val in range(step, max_val + 1, step):
        if val >= min_val:
            buckets.append(val)

    # Ensure max is included if not already
    if max_val not in buckets and max_val >= min_val:
        buckets.append(max_val)

    return sorted(set(buckets))


class LinearBucketing:
    """
    Linear bucketing manager for HPU graph warmup.

    Generates bucket shapes for prefill and decode phases:
    - Prefill: (batch_size, seq_len) - varying sequence lengths
    - Decode: (batch_size, num_blocks) - fixed seq_len=1, varying context

    Usage:
        bucketing = LinearBucketing(config)
        bucketing.initialize(num_kvcache_blocks)

        # During inference:
        bucket = bucketing.find_bucket(batch_size, seq_len, is_prefill=True)
        padded_input = pad_to_bucket(input, bucket)
    """

    def __init__(self, config):
        """
        Initialize bucketing with config parameters.

        Args:
            config: Config object with max_model_len, max_num_seqs, etc.
        """
        self.block_size = config.kvcache_block_size
        self.max_model_len = config.max_model_len
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens

        self.prefill_buckets: List[Tuple[int, int]] = []
        self.decode_buckets: List[Tuple[int, int]] = []

        self.num_kvcache_blocks: int = 0
        self._initialized = False

    def initialize(self, num_kvcache_blocks: int) -> None:
        """
        Initialize buckets after KV cache allocation.

        Must be called after allocate_kv_cache() so we know the
        actual number of KV cache blocks available.

        Args:
            num_kvcache_blocks: Number of KV cache blocks allocated
        """
        self.num_kvcache_blocks = num_kvcache_blocks
        self._generate_prefill_buckets()
        self._generate_decode_buckets()
        self._initialized = True

        logger.info(f"Generated {len(self.prefill_buckets)} prefill buckets: {self.prefill_buckets}")
        logger.info(f"Generated {len(self.decode_buckets)} decode buckets: {self.decode_buckets}")

    def _read_env_config(self, phase: str, dim: str, defaults: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Read bucket config from environment variables.

        Environment variables follow pattern:
            NANOVLLM_{PHASE}_{DIM}_BUCKET_{PARAM}

        Example:
            NANOVLLM_PROMPT_SEQ_BUCKET_MIN=128
            NANOVLLM_PROMPT_SEQ_BUCKET_STEP=128
            NANOVLLM_PROMPT_SEQ_BUCKET_MAX=2048

        Args:
            phase: 'PROMPT' or 'DECODE'
            dim: 'BS', 'SEQ', or 'BLOCK'
            defaults: (min, step, max) default values

        Returns:
            (min, step, max) tuple from env or defaults
        """
        prefix = f"NANOVLLM_{phase}_{dim}_BUCKET"
        min_val = int(os.environ.get(f"{prefix}_MIN", defaults[0]))
        step = int(os.environ.get(f"{prefix}_STEP", defaults[1]))
        max_val = int(os.environ.get(f"{prefix}_MAX", defaults[2]))

        logger.debug(f"{prefix}: min={min_val}, step={step}, max={max_val}")
        return (min_val, step, max_val)

    def _generate_prefill_buckets(self) -> None:
        """Generate prefill buckets: (batch_size, seq_len)."""
        # Batch size: ramp-up from 1, step at 4 for exponential growth [1, 2, 4, 8, 12, ...]
        bs_cfg = self._read_env_config("PROMPT", "BS", (1, 4, self.max_num_seqs))
        bs_range = warmup_range(*bs_cfg)

        # Sequence length: start at block_size, step at 2x block_size for fewer buckets
        # Example with block_size=128: [128, 256, 512, 768, 1024, ...]
        seq_cfg = self._read_env_config(
            "PROMPT", "SEQ",
            (self.block_size, self.block_size * 2, min(self.max_num_batched_tokens, self.max_model_len))
        )
        seq_range = warmup_range(*seq_cfg)

        for bs in bs_range:
            for seq_len in seq_range:
                # Filter: total tokens <= max_num_batched_tokens
                if bs * seq_len <= self.max_num_batched_tokens:
                    # Filter: seq_len <= max_model_len
                    if seq_len <= self.max_model_len:
                        self.prefill_buckets.append((bs, seq_len))

        self.prefill_buckets.sort()

    def _generate_decode_buckets(self) -> None:
        """Generate decode buckets: (batch_size, num_blocks)."""
        # Batch size: ramp-up from 1, step at 8 for exponential growth [1, 2, 4, 8, 16, 24, ...]
        bs_cfg = self._read_env_config("DECODE", "BS", (1, 8, self.max_num_seqs))
        bs_range = warmup_range(*bs_cfg)

        # Num blocks: context length in blocks
        # Max is limited by actual KV cache allocation
        max_blocks = min(
            self.num_kvcache_blocks,
            self.max_num_seqs * ((self.max_model_len + self.block_size - 1) // self.block_size)
        )
        # Step at 2x block_size for fewer buckets: [128, 256, 512, 768, ...]
        block_cfg = self._read_env_config("DECODE", "BLOCK", (self.block_size, self.block_size * 2, max_blocks))
        block_range = warmup_range(*block_cfg)

        for bs in bs_range:
            for num_blocks in block_range:
                # Filter: each sequence needs at least 1 block
                if bs <= num_blocks:
                    self.decode_buckets.append((bs, num_blocks))

        self.decode_buckets.sort()

    def find_bucket(self, batch_size: int, second_dim: int, is_prefill: bool) -> Tuple[int, int]:
        """
        Find smallest bucket that fits the request.

        Uses binary search for efficiency. If no pre-warmed bucket fits,
        generates a fallback bucket (which will trigger runtime compilation).

        Args:
            batch_size: Number of sequences in batch
            second_dim: seq_len for prefill, num_blocks for decode
            is_prefill: True for prefill phase, False for decode

        Returns:
            Bucket tuple (batch_size, seq_len/num_blocks)
        """
        if not self._initialized:
            raise RuntimeError("Bucketing not initialized. Call initialize() first.")

        buckets = self.prefill_buckets if is_prefill else self.decode_buckets
        target = (batch_size, second_dim)

        # Binary search for first candidate
        idx = bisect.bisect_left(buckets, target)

        # Find bucket where BOTH dimensions are >= target
        for i in range(idx, len(buckets)):
            if buckets[i][0] >= batch_size and buckets[i][1] >= second_dim:
                return buckets[i]

        # No pre-warmed bucket fits - generate fallback
        return self._generate_fallback(batch_size, second_dim, is_prefill)

    def _generate_fallback(self, batch_size: int, second_dim: int, is_prefill: bool) -> Tuple[int, int]:
        """
        Generate fallback bucket for shapes exceeding pre-warmed buckets.

        This will trigger runtime graph compilation (slow!) but ensures
        the request can still be processed.

        Args:
            batch_size: Requested batch size
            second_dim: Requested seq_len or num_blocks
            is_prefill: True for prefill phase

        Returns:
            New bucket tuple added to the bucket list
        """
        # Round up to reasonable alignment
        aligned_bs = ((batch_size + 7) // 8) * 8  # Align to 8
        aligned_dim = ((second_dim + self.block_size - 1) // self.block_size) * self.block_size

        bucket = (aligned_bs, aligned_dim)
        phase = "prefill" if is_prefill else "decode"
        logger.warning(
            f"Fallback {phase} bucket generated: {bucket} for request ({batch_size}, {second_dim}). "
            f"This triggers runtime graph compilation - consider increasing bucket max."
        )

        # Add to list for future requests
        if is_prefill:
            self.prefill_buckets.append(bucket)
            self.prefill_buckets.sort()
        else:
            self.decode_buckets.append(bucket)
            self.decode_buckets.sort()

        return bucket

    def get_prefill_buckets(self) -> List[Tuple[int, int]]:
        """Get all prefill buckets for warmup iteration."""
        return self.prefill_buckets.copy()

    def get_decode_buckets(self) -> List[Tuple[int, int]]:
        """Get all decode buckets for warmup iteration."""
        return self.decode_buckets.copy()

    @property
    def total_buckets(self) -> int:
        """Total number of buckets to warm up."""
        return len(self.prefill_buckets) + len(self.decode_buckets)
