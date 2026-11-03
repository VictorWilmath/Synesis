"""Which SteamVR device the camera sees at which keypoint.

This pairing is what makes calibration free. SteamVR already knows, in metres,
where the headset and controllers are, and the pose model already finds the
head and the wrists in the image. Every frame of normal use therefore yields up
to three 3D-to-2D correspondences, and enough of them pin down the camera.

Offsets are expressed in the device's local frame, not the world's. The
headset's tracking origin sits in front of the face, so the vector from it to
the head keypoint rotates as the user looks around. Treating that offset as
fixed in world space injects an error that grows with head rotation, and the
solver would absorb it as a camera-pose error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..skeleton import HEAD, LEFT_WRIST, RIGHT_WRIST


@dataclass(frozen=True, slots=True)
class Anchor:
    """A SteamVR device paired with the keypoint the camera sees it at."""

    name: str
    keypoint: int
    # Device-local vector from the tracking origin to the observed keypoint.
    # Refined by the calibrator; these are only starting values.
    local_offset: np.ndarray
    # The keypoint this device appears at if the image is mirrored, where the
    # pose model's left and right trade places.
    mirrored_keypoint: int


# Nominal offsets. The headset's origin is at the display, so the head keypoint
# is behind and slightly above it; a controller's origin is near the palm, so
# the wrist is a little further back along the handle. Both are refined during
# calibration, but a sane start keeps the optimiser out of local minima.
ANCHORS: tuple[Anchor, ...] = (
    Anchor("head", HEAD, np.array([0.0, 0.045, 0.10]), HEAD),
    Anchor("left_hand", LEFT_WRIST, np.array([0.0, 0.015, 0.055]), RIGHT_WRIST),
    Anchor("right_hand", RIGHT_WRIST, np.array([0.0, 0.015, 0.055]), LEFT_WRIST),
)

ANCHOR_BY_NAME: dict[str, Anchor] = {a.name: a for a in ANCHORS}
