"""The lifting interface shared by the geometric baseline and the trained model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..types import Keypoints2D, Skeleton3D, VRState


@dataclass(slots=True)
class LiftContext:
    """Everything a lifter may draw on beyond the current frame's keypoints."""

    intrinsics: np.ndarray
    bone_lengths: dict[tuple[int, int], float]
    height_m: float = 1.75
    # World-to-camera extrinsics, once calibration has solved them. Without
    # these the output is camera-relative and only good enough to look at.
    rotation_cw: np.ndarray | None = None
    translation_cw: np.ndarray | None = None
    vr: VRState | None = None
    # Fallback camera placement used before extrinsics exist.
    assumed_camera_height_m: float = 1.1
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return self.rotation_cw is not None and self.translation_cw is not None

    def up_in_camera(self) -> np.ndarray:
        """World up, expressed in camera space.

        Lets the lifter apply a gravity prior. Before extrinsics exist this
        assumes a level camera, where up is simply camera -y; that is close
        enough to be useful and no worse than having no prior at all.
        """
        if self.rotation_cw is None:
            return np.array([0.0, -1.0, 0.0])
        return np.asarray(self.rotation_cw, dtype=np.float64) @ np.array([0.0, 1.0, 0.0])


class Lifter(Protocol):
    """Turns 2D keypoints into a metric 3D skeleton in play space."""

    def __call__(self, keypoints: Keypoints2D, context: LiftContext) -> Skeleton3D | None: ...

    def reset(self) -> None: ...
