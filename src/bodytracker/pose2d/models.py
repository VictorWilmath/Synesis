"""RTMPose and YOLOX checkpoint URLs for the Halpe26 layout.

Halpe26 is the layout this project uses everywhere; see `bodytracker.skeleton`
for why. Accuracy figures are AP on Body8, as published by OpenMMLab.

RTMPose-m is the default: 76.7 AP, and roughly 1.5 ms per frame on an RTX 2070,
which leaves the GPU almost entirely free for VRChat to render with.
"""

from __future__ import annotations

from dataclasses import dataclass

_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk"


@dataclass(frozen=True, slots=True)
class ModelPreset:
    detector: str
    detector_input_size: tuple[int, int]
    pose: str
    pose_input_size: tuple[int, int]
    description: str


PRESETS: dict[str, ModelPreset] = {
    # For running the pose model on the CPU, leaving the GPU to VRChat.
    "lightweight": ModelPreset(
        detector=f"{_BASE}/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        detector_input_size=(416, 416),
        pose=f"{_BASE}/rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
        pose_input_size=(192, 256),
        description="YOLOX-tiny + RTMPose-s, 72.0 AP",
    ),
    "balanced": ModelPreset(
        detector=f"{_BASE}/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
        detector_input_size=(640, 640),
        pose=f"{_BASE}/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        pose_input_size=(192, 256),
        description="YOLOX-m + RTMPose-m, 76.7 AP",
    ),
    "performance": ModelPreset(
        detector=f"{_BASE}/yolox_x_8xb8-300e_humanart-a39d44ed.zip",
        detector_input_size=(640, 640),
        pose=f"{_BASE}/rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip",
        pose_input_size=(192, 256),
        description="YOLOX-x + RTMPose-l, 78.4 AP",
    ),
}

DEFAULT_PRESET = "balanced"


def get_preset(name: str) -> ModelPreset:
    if name not in PRESETS:
        raise ValueError(f"unknown pose2d mode '{name}', expected one of {sorted(PRESETS)}")
    return PRESETS[name]
