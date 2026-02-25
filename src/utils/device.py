"""
Device Detection Utility
=========================
Auto-detects the best available compute device:
  - Apple Silicon Mac → MPS (Metal Performance Shaders)
  - NVIDIA GPU         → CUDA
  - Fallback           → CPU
"""

import torch
import platform
import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Return the best available PyTorch device string."""
    if torch.cuda.is_available():
        device = "cuda:0"
        name = torch.cuda.get_device_name(0)
        logger.info("Using CUDA device: %s", name)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple MPS (Metal) on %s", platform.processor())
    else:
        device = "cpu"
        logger.info("Using CPU (%s)", platform.processor())
    return device


def get_device_info() -> dict:
    """Return detailed device information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_mem / 1e9, 1
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device"] = "mps"
        info["gpu_name"] = "Apple Metal GPU"
    else:
        info["device"] = "cpu"
    return info
