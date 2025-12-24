"""HPU platform implementation for Intel Gaudi."""

from nanovllm.platforms.hpu.platform import HpuPlatform
from nanovllm.platforms.hpu.memory import (
    HpuMemoryProfiler,
    is_fake_hpu,
    format_bytes,
)
from nanovllm.platforms.hpu.bucketing import (
    LinearBucketing,
    warmup_range,
)
from nanovllm.platforms.hpu.attention import (
    HpuAttentionMetadata,
    HpuAttentionBackend,
    ModuleFusedSDPA,
    batch2block,
    block2batch,
    grouped_max,
    pipelined_pa,
)

__all__ = [
    "HpuPlatform",
    "HpuMemoryProfiler",
    "is_fake_hpu",
    "format_bytes",
    "LinearBucketing",
    "warmup_range",
    # Attention
    "HpuAttentionMetadata",
    "HpuAttentionBackend",
    "ModuleFusedSDPA",
    "batch2block",
    "block2batch",
    "grouped_max",
    "pipelined_pa",
]
