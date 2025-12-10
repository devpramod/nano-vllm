import os
from dataclasses import dataclass, field
from typing import Literal

import torch
from transformers import AutoConfig


# Type aliases for device and backend options
DeviceType = Literal["cuda", "hpu", "auto"]
DistributedBackend = Literal["nccl", "hccl", "auto"]


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = -1  # -1 means auto-select based on device
    num_kvcache_blocks: int = -1

    # HPU-specific fields
    device_type: DeviceType = "auto"
    distributed_backend: DistributedBackend = "auto"
    lazy_mode: bool = True  # HPU lazy mode (ignored for CUDA)
    use_fused_sdpa: bool = True  # Use FusedSDPA on HPU (ignored for CUDA)
    dtype: torch.dtype = field(default=torch.bfloat16)

    def __post_init__(self):

        # Auto-detect device type if set to 'auto'
        if self.device_type == "auto":
            self.device_type = self._detect_device_type()

        # Auto-select distributed backend based on device type
        if self.distributed_backend == "auto":
            self.distributed_backend = self._get_default_backend()

        # Auto-select block size based on device type if not specified
        if self.kvcache_block_size == -1:
            self.kvcache_block_size = 128 if self.device_type == "hpu" else 256

        # Device-specific validation
        self._validate_for_device()

        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len

    def _detect_device_type(self) -> DeviceType:
        """Auto-detect the available device type."""
        # Check for HPU first (more specific)
        try:
            import habana_frameworks.torch as htorch  # noqa: F401

            if torch.hpu.is_available():
                return "hpu"
        except ImportError:
            pass

        # Fall back to CUDA
        if torch.cuda.is_available():
            return "cuda"

        # Default to CUDA even if not available (will fail later with clearer error)
        return "cuda"

    def _get_default_backend(self) -> DistributedBackend:
        """Get the default distributed backend for the current device type."""
        if self.device_type == "hpu":
            return "hccl"
        return "nccl"

    def _validate_for_device(self) -> None:
        """Validate configuration constraints for the target device."""
        if self.device_type == "hpu":
            self._validate_hpu_config()
        else:
            self._validate_cuda_config()

    def _validate_hpu_config(self) -> None:
        """Validate HPU-specific configuration constraints."""
        # HPU requires block size to be divisible by 128
        if self.kvcache_block_size % 128 != 0:
            raise ValueError(
                f"KV cache block size must be divisible by 128 for HPU, "
                f"got {self.kvcache_block_size}"
            )

        # HPU only supports bfloat16
        if self.dtype != torch.bfloat16:
            raise ValueError(
                f"HPU only supports bfloat16 dtype, got {self.dtype}. "
                f"Set dtype=torch.bfloat16 for HPU."
            )

    def _validate_cuda_config(self) -> None:
        """Validate CUDA-specific configuration constraints."""
        # CUDA block size should be divisible by 256 for Flash Attention
        if self.kvcache_block_size % 256 != 0:
            raise ValueError(
                f"KV cache block size must be divisible by 256 for CUDA, "
                f"got {self.kvcache_block_size}"
            )

    def get_dtype(self) -> torch.dtype:
        """Get the dtype for model computations."""
        return self.dtype
