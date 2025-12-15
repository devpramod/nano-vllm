"""HPU platform implementation for Intel Gaudi."""

from nanovllm.platforms.hpu.platform import HpuPlatform
from nanovllm.platforms.hpu.memory import (
    HpuMemoryProfiler,
    is_fake_hpu,
    format_bytes,
)

__all__ = ["HpuPlatform", "HpuMemoryProfiler", "is_fake_hpu", "format_bytes"]
