"""HMD-anchored lifting: the reason this approach can beat a plain webcam tracker.

A single camera cannot recover scale. Two people of different sizes at
different distances produce identical images, so every monocular tracker has to
guess, usually by assuming a body height. The guess is wrong by a few percent,
and a few percent of three metres is enough that your feet do not touch the
floor.

A headset removes the guess. SteamVR reports, in metres, exactly where the
user's head is. Once the camera pose is known, that fixes the depth of one
joint outright, and propagating from there gives a metrically correct skeleton
rather than a scaled one.

The lateral disagreement between where the HMD says the head is and where the
pose model draws it is not thrown away: it is the most useful health signal the
system has. It rises when calibration drifts, when the camera is knocked, or
when the pose model latches onto the wrong person, and none of those are
otherwise easy to detect at runtime.
"""

from __future__ import annotations

import logging

import numpy as np

from ..calib.anchors import ANCHOR_BY_NAME
from ..geom import camera_from_play, play_from_camera
from ..skeleton import HEAD, LEFT_WRIST, RIGHT_WRIST, ROOT
from ..types import Keypoints2D, Skeleton3D, VRState
from .base import LiftContext
from .geometric import GeometricLifter
from .kinematics import enforce_bone_lengths, lift_by_bone_lengths

log = logging.getLogger(__name__)

_ANCHOR_KEYPOINT = {"head": HEAD, "left_hand": LEFT_WRIST, "right_hand": RIGHT_WRIST}


def anchor_points_play(
    vr: VRState,
    offsets: dict[str, np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Where SteamVR says each anchored keypoint is, in play space."""
    out: dict[int, np.ndarray] = {}
    for name, pose in vr.anchors().items():
        anchor = ANCHOR_BY_NAME.get(name)
        if anchor is None:
            continue
        offset = anchor.local_offset if offsets is None else offsets.get(name, anchor.local_offset)
        position = np.asarray(pose.position) + np.asarray(pose.rotation) @ offset
        out[_ANCHOR_KEYPOINT[name]] = position
    return out


class AnchoredLifter:
    """Bone-length lifting with the head pinned to the headset's depth.

    Falls back to the unanchored behaviour whenever the headset or the camera
    calibration is unavailable, so the tracker degrades rather than stops.
    """

    def __init__(
        self,
        min_score: float = 0.3,
        *,
        offsets: dict[str, np.ndarray] | None = None,
        max_head_residual_m: float = 0.35,
    ) -> None:
        self.min_score = min_score
        self.offsets = offsets
        # Beyond this the pose model is almost certainly not looking at the
        # headset wearer, and anchoring to it would drag the skeleton away.
        self.max_head_residual_m = max_head_residual_m

        self._previous_depths: np.ndarray | None = None
        self.head_residual_m: float | None = None
        self.wrist_residual_m: float | None = None
        self.anchored = False
        self.rejected_anchors = 0

    def reset(self) -> None:
        self._previous_depths = None
        self.head_residual_m = None
        self.wrist_residual_m = None
        self.anchored = False

    def __call__(self, keypoints: Keypoints2D, context: LiftContext) -> Skeleton3D | None:
        anchors = self._usable_anchors(context)
        root_joint, root_depth = self._pin(anchors, context)

        camera_xyz = lift_by_bone_lengths(
            keypoints.xy,
            keypoints.scores,
            context.intrinsics,
            context.bone_lengths,
            min_score=self.min_score,
            root_depth=root_depth,
            depth_prior=self._previous_depths,
            root_joint=root_joint,
            up_camera=context.up_in_camera(),
            facing_camera=self._facing_hint(context),
        )
        if camera_xyz is None:
            self.anchored = False
            return None

        camera_xyz = enforce_bone_lengths(camera_xyz, context.bone_lengths)
        self._previous_depths = camera_xyz[:, 2].copy()

        if context.calibrated:
            play_xyz = play_from_camera(camera_xyz, context.rotation_cw, context.translation_cw)
        else:
            # No extrinsics yet, so fall back to the baseline's rough placement.
            play_xyz = GeometricLifter._uncalibrated_to_play(camera_xyz, context)
            play_xyz = GeometricLifter._place_on_floor(play_xyz)

        self._measure_residuals(play_xyz, anchors)

        return Skeleton3D(
            xyz=play_xyz.astype(np.float32),
            scores=np.asarray(keypoints.scores, dtype=np.float32).copy(),
            space="play",
            timestamp=keypoints.timestamp,
        )

    @staticmethod
    def _facing_hint(context: LiftContext) -> np.ndarray | None:
        """Roughly which way the body faces, taken from the headset.

        Resolves the left/right mirror ambiguity in the shoulder and hip lines,
        which a single camera cannot settle on its own: turned slightly left
        and turned slightly right project to nearly the same silhouette.

        The head is not the torso, so this is only a hint. It is used to choose
        between two discrete options, not as a measurement, and being wrong by
        even sixty degrees still picks the right one.
        """
        if context.vr is None or not context.calibrated:
            return None
        head = context.vr.head
        if head is None or not head.valid:
            return None

        forward_play = np.asarray(head.rotation, dtype=np.float64) @ np.array([0.0, 0.0, -1.0])
        # Flatten to the horizontal plane; looking up or down says nothing
        # about which way the body is turned.
        forward_play[1] = 0.0
        norm = float(np.linalg.norm(forward_play))
        if norm < 1e-6:
            return None
        return np.asarray(context.rotation_cw, dtype=np.float64) @ (forward_play / norm)

    def _usable_anchors(self, context: LiftContext) -> dict[int, np.ndarray]:
        if context.vr is None or not context.calibrated:
            return {}
        return anchor_points_play(context.vr, self.offsets)

    def _pin(
        self, anchors: dict[int, np.ndarray], context: LiftContext
    ) -> tuple[int, float | None]:
        """Choose the joint to pin and the depth to pin it at."""
        head_play = anchors.get(HEAD)
        if head_play is None:
            self.anchored = False
            return ROOT, None

        head_camera = camera_from_play(
            head_play.reshape(1, 3), context.rotation_cw, context.translation_cw
        )[0]
        if head_camera[2] <= 0.2:
            # The headset is behind the camera, so calibration is wrong.
            self.anchored = False
            self.rejected_anchors += 1
            return ROOT, None

        # Guard against anchoring to a bystander: if the previous frame's head
        # was far from where the headset says it is, do not pin.
        if self.head_residual_m is not None and self.head_residual_m > self.max_head_residual_m:
            self.anchored = False
            self.rejected_anchors += 1
            return ROOT, None

        self.anchored = True
        return HEAD, float(head_camera[2])

    def _measure_residuals(self, play_xyz: np.ndarray, anchors: dict[int, np.ndarray]) -> None:
        """Compare the solved joints against the devices that were not pinned.

        The wrists are never pinned, so their residual is a genuine held-out
        check on the whole chain: intrinsics, extrinsics, keypoints and bone
        lengths all have to be right for it to be small.
        """
        head_play = anchors.get(HEAD)
        self.head_residual_m = (
            float(np.linalg.norm(play_xyz[HEAD] - head_play)) if head_play is not None else None
        )

        wrist_errors = [
            float(np.linalg.norm(play_xyz[joint] - point))
            for joint, point in anchors.items()
            if joint in (LEFT_WRIST, RIGHT_WRIST)
        ]
        self.wrist_residual_m = float(np.mean(wrist_errors)) if wrist_errors else None

    def diagnostics(self) -> dict[str, float | bool | None]:
        return {
            "anchored": self.anchored,
            "head_residual_m": self.head_residual_m,
            "wrist_residual_m": self.wrist_residual_m,
            "rejected_anchors": self.rejected_anchors,
        }
