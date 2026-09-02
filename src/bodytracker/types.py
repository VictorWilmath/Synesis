"""Data structures passed between pipeline stages.

Every array carries its coordinate space in the field name or docstring. The
spaces in play are:

``image``
    Pixels, origin top-left, +x right, +y down.
``camera``
    OpenCV convention: +x right, +y down, +z forward (into the scene), metres.
``play``
    SteamVR standing-universe space: right-handed, +y up, -z forward, metres.
    This is also VRChat's tracking space, which is why the tracker output needs
    only a handedness flip and no alignment hack.
``unity``
    Left-handed, +y up, +z forward, metres. What actually goes on the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .skeleton import NUM_KEYPOINTS, TrackerRole


@dataclass(slots=True)
class Keypoints2D:
    """Halpe26 keypoints for a single frame, in image space."""

    xy: np.ndarray  # (26, 2) float32, pixels
    scores: np.ndarray  # (26,) float32 in [0, 1]
    bbox: np.ndarray | None = None  # (4,) float32 xyxy, the crop they came from
    timestamp: float = 0.0  # seconds, monotonic clock

    def __post_init__(self) -> None:
        if self.xy.shape != (NUM_KEYPOINTS, 2):
            raise ValueError(f"expected xy of shape ({NUM_KEYPOINTS}, 2), got {self.xy.shape}")
        if self.scores.shape != (NUM_KEYPOINTS,):
            raise ValueError(
                f"expected scores of shape ({NUM_KEYPOINTS},), got {self.scores.shape}"
            )

    def valid(self, min_score: float) -> np.ndarray:
        """(26,) bool mask of keypoints confident enough to use."""
        return self.scores >= min_score


@dataclass(slots=True)
class Skeleton3D:
    """Halpe26 joints in metric 3D."""

    xyz: np.ndarray  # (26, 3) float32
    scores: np.ndarray  # (26,) float32, carried through from the 2D stage
    space: str = "play"
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.xyz.shape != (NUM_KEYPOINTS, 3):
            raise ValueError(f"expected xyz of shape ({NUM_KEYPOINTS}, 3), got {self.xyz.shape}")

    def copy(self) -> Skeleton3D:
        return Skeleton3D(self.xyz.copy(), self.scores.copy(), self.space, self.timestamp)


@dataclass(slots=True)
class DevicePose:
    """A 6DoF pose read from SteamVR, in play space."""

    position: np.ndarray  # (3,) float32, metres
    rotation: np.ndarray  # (3, 3) float32, rotation matrix
    valid: bool = True
    timestamp: float = 0.0


@dataclass(slots=True)
class VRState:
    """The SteamVR devices we use as anchors. Any of them may be missing."""

    head: DevicePose | None = None
    left_hand: DevicePose | None = None
    right_hand: DevicePose | None = None
    timestamp: float = 0.0

    def anchors(self) -> dict[str, DevicePose]:
        """Valid anchors keyed by the name the calibrator uses."""
        out: dict[str, DevicePose] = {}
        for name, pose in (
            ("head", self.head),
            ("left_hand", self.left_hand),
            ("right_hand", self.right_hand),
        ):
            if pose is not None and pose.valid:
                out[name] = pose
        return out


@dataclass(slots=True)
class TrackerTarget:
    """One VRChat tracker slot's worth of output."""

    role: TrackerRole
    position: np.ndarray  # (3,) float32, play space, metres
    rotation: np.ndarray  # (3, 3) float32, play space
    valid: bool = True
    # True when this is a held-over value from an earlier frame rather than a
    # fresh estimate. The sender still transmits it; VRChat prefers a slightly
    # stale tracker to one that teleports.
    stale: bool = False


@dataclass(slots=True)
class FrameResult:
    """Everything one pipeline iteration produced, for the overlay and recorder."""

    timestamp: float
    keypoints2d: Keypoints2D | None = None
    skeleton3d: Skeleton3D | None = None
    vr: VRState | None = None
    targets: list[TrackerTarget] = field(default_factory=list)
    # Wall-clock milliseconds per stage, for the latency budget.
    timings: dict[str, float] = field(default_factory=dict)
    # Stage-specific health signals, e.g. the lifter's HMD residual.
    diagnostics: dict[str, object] = field(default_factory=dict)
