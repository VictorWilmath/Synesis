"""Small Windows compatibility helpers shared by ONNX-backed components."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_cuda_dlls() -> None:
    """Expose CUDA 11.8 and cuDNN DLLs installed inside the active venv.

    ONNX Runtime's CUDA wheel keeps the machine PATH untouched. PyTorch and
    NVIDIA's Windows wheels keep their dependency DLLs under site-packages, so
    add those folders before either rtmlib or the neural lifter opens ONNX.
    """
    if os.name != "nt":
        return
    site = Path(sys.executable).parent.parent / "Lib" / "site-packages"
    folders = (
        site / "torch" / "lib",
        site / "nvidia" / "cudnn" / "bin",
        site / "nvidia" / "cublas" / "bin",
        site / "nvidia" / "cuda_nvrtc" / "bin",
    )
    paths = [str(folder) for folder in folders if folder.is_dir()]
    if paths:
        os.environ["PATH"] = os.pathsep.join((*paths, os.environ.get("PATH", "")))
