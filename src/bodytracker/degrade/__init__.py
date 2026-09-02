"""Making clean renders look like webcam footage.

The gap between a BEDLAM render and a 720p USB webcam is not resolution, it is
noise, MJPG compression, motion blur, and an ISP doing things nobody asked for.
Closing that gap in the training data is what makes a model trained on
synthetic images work on real ones.
"""

from .pipeline import (
    ClipDegrader,
    ClipParameters,
    DegradationConfig,
    advance,
    degrade_frame,
    sample_clip,
)
from .profile import (
    CameraProfile,
    estimate_drift,
    estimate_noise,
    linear_to_srgb,
    profile_frames,
    srgb_to_linear,
)

__all__ = [
    "CameraProfile",
    "ClipDegrader",
    "ClipParameters",
    "DegradationConfig",
    "advance",
    "degrade_frame",
    "estimate_drift",
    "estimate_noise",
    "linear_to_srgb",
    "profile_frames",
    "sample_clip",
    "srgb_to_linear",
]
