"""
Platform abstraction layer for nano-vllm.

This module provides a plugin architecture for supporting different hardware
backends (CUDA, HPU, etc.). Platforms are selected via:

1. Explicit: NANOVLLM_PLATFORM environment variable
2. Auto-detect: First available platform (CUDA -> HPU -> error)

Usage:
    from nanovllm.platforms import get_platform

    platform = get_platform()  # Auto-detect
    platform = get_platform("cuda")  # Explicit

    platform.set_device(0)
    free, total = platform.get_memory_info()
"""

from nanovllm.platforms.base import Platform

__all__ = ["Platform", "get_platform", "register_platform"]

# Platform registry
_PLATFORMS: dict[str, type[Platform]] = {}

# Cached platform instance
_current_platform: Platform | None = None


def register_platform(name: str, platform_cls: type[Platform]) -> None:
    """
    Register a platform implementation.

    Args:
        name: Platform name (e.g., "cuda", "hpu").
        platform_cls: Platform class to register.
    """
    _PLATFORMS[name] = platform_cls


def _auto_detect_platform() -> str:
    """
    Auto-detect the best available platform.

    Returns:
        Platform name.

    Raises:
        RuntimeError: If no platform is available.
    """
    import torch

    # Try CUDA first
    if torch.cuda.is_available():
        return "cuda"

    # Try HPU
    try:
        if torch.hpu.is_available():
            return "hpu"
    except AttributeError:
        pass

    raise RuntimeError(
        "No platform available. Install CUDA or run in Gaudi container."
    )


def get_platform(name: str | None = None) -> Platform:
    """
    Get a platform instance.

    Args:
        name: Platform name. If None, uses NANOVLLM_PLATFORM env var
              or auto-detects.

    Returns:
        Platform instance.

    Raises:
        ValueError: If platform name is unknown.
        RuntimeError: If no platform is available.
    """
    global _current_platform

    import os

    if name is None:
        name = os.environ.get("NANOVLLM_PLATFORM")

    if name is None:
        name = _auto_detect_platform()

    if name not in _PLATFORMS:
        available = list(_PLATFORMS.keys())
        raise ValueError(
            f"Unknown platform '{name}'. Available: {available}"
        )

    # Return cached instance if same platform
    if _current_platform is not None and _current_platform.device_name == name:
        return _current_platform

    _current_platform = _PLATFORMS[name]()
    return _current_platform


# Discover and register platforms via entry points
def _discover_platforms():
    """
    Discover platform implementations via entry points.

    This allows external packages to register platforms by declaring
    entry points in their pyproject.toml:

        [project.entry-points."nanovllm.platforms"]
        hpu = "my_package:HpuPlatform"
    """
    import sys

    if sys.version_info >= (3, 10):
        from importlib.metadata import entry_points
        eps = entry_points(group="nanovllm.platforms")
    else:
        from importlib.metadata import entry_points
        eps = entry_points().get("nanovllm.platforms", [])

    for ep in eps:
        try:
            platform_cls = ep.load()
            register_platform(ep.name, platform_cls)
        except Exception:
            pass  # Platform dependencies not available


_discover_platforms()
