"""2D keypoint estimation and bounding-box tracking."""

from .models import DEFAULT_PRESET, PRESETS, ModelPreset, get_preset
from .tracking import (
    bbox_from_keypoints,
    clip_bbox,
    expand_to_aspect,
    iou,
    select_person,
    valid_ratio,
)

__all__ = [
    "DEFAULT_PRESET",
    "PRESETS",
    "ModelPreset",
    "bbox_from_keypoints",
    "clip_bbox",
    "expand_to_aspect",
    "get_preset",
    "iou",
    "select_person",
    "valid_ratio",
]


def __getattr__(name: str):
    # Deferred so that importing this package does not pull in onnxruntime.
    if name == "Pose2DEstimator":
        from .estimator import Pose2DEstimator

        return Pose2DEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
