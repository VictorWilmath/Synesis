"""The Phase 1 baseline lifter: bone lengths and a pinhole camera, nothing learned.

This exists to get the pipeline end-to-end and, more importantly, to be the
number that later work has to beat. Its weaknesses are known and expected:

- Depth ambiguity is resolved by assuming limbs are closer to fronto-parallel
  than not, so a leg kicked toward the camera can invert.
- Absolute depth comes from apparent bone size, which is only as good as the
  height prior and the focal length.
- There is no temporal reasoning at all, so occluded joints snap back.

Every one of those is addressed either by the HMD anchor (Phase 2) or by the
trained lifter (Phase 4).
"""

from __future__ import annotations

import numpy as np

from ..geom import play_from_camera
from ..skeleton import LEFT_ANKLE, LEFT_HEEL, RIGHT_ANKLE, RIGHT_HEEL
from ..types import Keypoints2D, Skeleton3D
from .base import LiftContext
from .kinematics import enforce_bone_lengths, lift_by_bone_lengths


class GeometricLifter:
    """Lifts by propagating bone-length constraints along the kinematic tree."""

    def __init__(self, min_score: float = 0.3, ground: bool = True) -> None:
        self.min_score = min_score
        self.ground = ground
        # Previous frame's camera-space depths, used to break the two-root
        # ambiguity consistently instead of letting foreshortened limbs flip.
        self._previous_depths: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_depths = None

    def __call__(self, keypoints: Keypoints2D, context: LiftContext) -> Skeleton3D | None:
        camera_xyz = lift_by_bone_lengths(
            keypoints.xy,
            keypoints.scores,
            context.intrinsics,
            context.bone_lengths,
            min_score=self.min_score,
            depth_prior=self._previous_depths,
            up_camera=context.up_in_camera(),
        )
        if camera_xyz is None:
            return None

        camera_xyz = enforce_bone_lengths(camera_xyz, context.bone_lengths)
        self._previous_depths = camera_xyz[:, 2].copy()

        if context.calibrated:
            play_xyz = play_from_camera(camera_xyz, context.rotation_cw, context.translation_cw)
        else:
            play_xyz = self._uncalibrated_to_play(camera_xyz, context)

        if self.ground:
            play_xyz = self._place_on_floor(play_xyz)

        return Skeleton3D(
            xyz=play_xyz.astype(np.float32),
            scores=keypoints.scores.copy(),
            space="play",
            timestamp=keypoints.timestamp,
        )

    @staticmethod
    def _uncalibrated_to_play(camera_xyz: np.ndarray, context: LiftContext) -> np.ndarray:
        """Guess a play-space pose before extrinsics are known.

        Assumes a level camera at a nominal height looking down its own +z.
        Camera space is +y down, play space is +y up and -z forward, so this is
        a y and z flip plus a height offset. Good enough to see whether the
        pipeline runs; not good enough to track with.
        """
        play = np.empty_like(camera_xyz)
        play[:, 0] = camera_xyz[:, 0]
        play[:, 1] = context.assumed_camera_height_m - camera_xyz[:, 1]
        play[:, 2] = -camera_xyz[:, 2]
        return play

    @staticmethod
    def _place_on_floor(play_xyz: np.ndarray) -> np.ndarray:
        """Drop the skeleton so its lowest foot point rests on y = 0.

        Absolute height is the least reliable output of the uncalibrated path,
        and a floating or sunken avatar reads as far more broken than one whose
        depth is slightly off.
        """
        foot_indices = [LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL]
        lowest = float(np.min(play_xyz[foot_indices, 1]))
        out = play_xyz.copy()
        out[:, 1] -= lowest
        return out
