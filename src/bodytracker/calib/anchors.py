"""Correspondences between SteamVR devices and observed 2D keypoints.

This is what makes checkerboard-free extrinsic calibration possible. SteamVR
already knows, in metres, where the headset and controllers are. The camera can
see the head and the wrists. Each frame therefore yields up to three
3D-to-2D correspondences for free, and enough of them over a varied set of
poses pins down the camera's position and orientation in play space.

Two details matter:

*Offsets are expressed in the device's local frame, not the world's.* The
headset's tracking origin sits in front of the face, so the vector from it to
the head keypoint rotates as the user looks around. Treating that offset as
fixed in world space injects an error that grows with head rotation, and PnP
absorbs it as a camera-pose error.

*Samples must be spatially varied.* A hundred frames of somebody standing still
is one correspondence repeated a hundred times, and PnP on it is
ill-conditioned. The buffer below rejects samples that are too close to ones it
already has, so the user is forced to actually cover the space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..skeleton import HEAD, LEFT_WRIST, RIGHT_WRIST
from ..types import Keypoints2D, VRState


@dataclass(frozen=True, slots=True)
class Anchor:
    """A SteamVR device paired with the keypoint the camera sees it at."""

    name: str
    keypoint: int
    # Device-local vector from the tracking origin to the observed keypoint.
    # Refined by the calibrator; these are only starting values.
    local_offset: np.ndarray


# Nominal offsets. The headset's origin is at the display, so the head keypoint
# is behind and slightly above it; a controller's origin is near the palm, so
# the wrist is a little further back along the handle. Both are refined during
# calibration, but a sane start keeps the optimiser out of local minima.
ANCHORS: tuple[Anchor, ...] = (
    Anchor("head", HEAD, np.array([0.0, 0.045, 0.10])),
    Anchor("left_hand", LEFT_WRIST, np.array([0.0, 0.015, 0.055])),
    Anchor("right_hand", RIGHT_WRIST, np.array([0.0, 0.015, 0.055])),
)

ANCHOR_BY_NAME: dict[str, Anchor] = {a.name: a for a in ANCHORS}


@dataclass(slots=True)
class Correspondence:
    """One device observation: where SteamVR says it is, and where it appears."""

    anchor: str
    device_position: np.ndarray  # (3,) play space
    device_rotation: np.ndarray  # (3, 3) play space
    pixel: np.ndarray  # (2,)
    score: float
    timestamp: float

    def world_point(self, local_offset: np.ndarray) -> np.ndarray:
        """The anchor point in play space, offset in the device's own frame."""
        return self.device_position + self.device_rotation @ local_offset


@dataclass(slots=True)
class CorrespondenceBuffer:
    """Accumulates well-spread correspondences for a PnP solve."""

    min_spacing_m: float = 0.05
    min_score: float = 0.5
    max_samples: int = 600
    allowed_anchors: tuple[str, ...] = ("head", "left_hand", "right_hand")
    samples: list[Correspondence] = field(default_factory=list)
    _head_positions: list[np.ndarray] = field(default_factory=list)
    rejected_close: int = 0
    rejected_unconfident: int = 0

    def add(self, vr: VRState, keypoints: Keypoints2D) -> int:
        """Record correspondences from one frame. Returns how many were kept."""
        if len(self.samples) >= self.max_samples:
            return 0

        head = vr.head
        if head is None or not head.valid:
            return 0

        # Novelty gate on head position, so standing still stops adding data.
        position = np.asarray(head.position, dtype=np.float64)
        if any(
            float(np.linalg.norm(position - existing)) < self.min_spacing_m
            for existing in self._head_positions
        ):
            self.rejected_close += 1
            return 0

        poses = vr.anchors()
        added = 0
        for anchor in ANCHORS:
            if anchor.name not in self.allowed_anchors:
                continue
            pose = poses.get(anchor.name)
            if pose is None:
                continue
            score = float(keypoints.scores[anchor.keypoint])
            if score < self.min_score:
                self.rejected_unconfident += 1
                continue
            self.samples.append(
                Correspondence(
                    anchor=anchor.name,
                    device_position=np.asarray(pose.position, dtype=np.float64).copy(),
                    device_rotation=np.asarray(pose.rotation, dtype=np.float64).copy(),
                    pixel=np.asarray(keypoints.xy[anchor.keypoint], dtype=np.float64).copy(),
                    score=score,
                    timestamp=keypoints.timestamp,
                )
            )
            added += 1

        if added:
            self._head_positions.append(position)
        return added

    def clear(self) -> None:
        self.samples.clear()
        self._head_positions.clear()
        self.rejected_close = 0
        self.rejected_unconfident = 0

    def __len__(self) -> int:
        return len(self.samples)

    def counts(self) -> dict[str, int]:
        out = {a.name: 0 for a in ANCHORS}
        for sample in self.samples:
            out[sample.anchor] += 1
        return out

    def coverage(self) -> float:
        """Spatial extent of the sampled head positions, in metres.

        A PnP solve from correspondences spread over a few centimetres is
        numerically hopeless regardless of how many there are, so the
        calibrator checks this before trusting a result.
        """
        if len(self._head_positions) < 2:
            return 0.0
        positions = np.array(self._head_positions)
        return float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))


def build_point_arrays(
    samples: list[Correspondence],
    offsets: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ``(object_points, image_points)`` for a PnP solve."""
    if offsets is None:
        offsets = {a.name: a.local_offset for a in ANCHORS}

    object_points = np.array([s.world_point(offsets[s.anchor]) for s in samples], dtype=np.float64)
    image_points = np.array([s.pixel for s in samples], dtype=np.float64)
    return object_points, image_points


def pack_offsets(offsets: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([offsets[a.name] for a in ANCHORS])


def unpack_offsets(vector: np.ndarray) -> dict[str, np.ndarray]:
    return {a.name: np.asarray(vector[i * 3 : i * 3 + 3]) for i, a in enumerate(ANCHORS)}
