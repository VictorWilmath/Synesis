"""Tests for HMD-anchored lifting.

The point of the whole project is here. A monocular tracker has to guess the
subject's size, and the guess is always somewhat wrong, which puts the whole
skeleton at the wrong distance. These tests check that pinning the head to the
headset removes that error, and that the anchoring degrades safely when the
headset, the calibration or the pose model lets it down.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.geom.spaces import intrinsics_from_fov
from bodytracker.lift import AnchoredLifter, GeometricLifter, LiftContext
from bodytracker.lift.kinematics import bone_lengths_from_height
from bodytracker.skeleton import (
    DEPTH_COPY_JOINTS,
    HEAD,
    HIP,
    LEFT_ANKLE,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_WRIST,
)
from bodytracker.tools.scene import build_scene

INTRINSICS = intrinsics_from_fov(1280, 720, 62.0)
TRUE_HEIGHT = 1.90
CAMERA_POSITION = np.array([0.4, 1.2, 2.8])


@pytest.fixture(scope="module")
def scene():
    return build_scene(
        intrinsics=INTRINSICS,
        camera_position=CAMERA_POSITION,
        height_m=TRUE_HEIGHT,
        frames=40,
    )


def make_context(scene, assumed_height: float, *, calibrated: bool = True) -> LiftContext:
    return LiftContext(
        intrinsics=INTRINSICS,
        bone_lengths=bone_lengths_from_height(assumed_height),
        height_m=assumed_height,
        rotation_cw=scene.rotation_cw if calibrated else None,
        translation_cw=scene.translation_cw if calibrated else None,
    )


def run(lifter, scene, context, frames: int | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Lift each frame, returning (solved, truth) pairs in play space."""
    out = []
    for frame in scene.frames[:frames]:
        context.vr = frame.vr
        skeleton = lifter(frame.keypoints, context)
        if skeleton is not None:
            out.append((np.asarray(skeleton.xyz, dtype=np.float64), frame.play_xyz))
    return out


SOLVED_JOINTS = [i for i in range(26) if i not in DEPTH_COPY_JOINTS]


def median_joint_error(pairs) -> float:
    """Median over frames of the mean joint error, in metres.

    Face keypoints are excluded because they inherit the head's depth by
    design rather than being solved, so including them would measure a
    deliberate approximation instead of the lifter's accuracy.
    """
    return float(
        np.median(
            [np.linalg.norm(s[SOLVED_JOINTS] - t[SOLVED_JOINTS], axis=1).mean() for s, t in pairs]
        )
    )


class TestAnchoringAccuracy:
    def test_recovers_the_pose_when_everything_is_known(self, scene):
        """Correct bone lengths, correct calibration: this should be near exact."""
        lifter = AnchoredLifter()
        pairs = run(lifter, scene, make_context(scene, TRUE_HEIGHT))
        assert len(pairs) == len(scene.frames)
        assert median_joint_error(pairs) < 0.01

    def test_no_joint_is_grossly_wrong(self, scene):
        """Catches a limb flipping to the wrong depth root.

        A flip is not a small error, it puts a shoulder on the wrong side of
        the neck or a foot pointing backwards, so it needs its own bound rather
        than being averaged away.
        """
        pairs = run(AnchoredLifter(), scene, make_context(scene, TRUE_HEIGHT))
        worst = max(
            float(np.max(np.linalg.norm(s[SOLVED_JOINTS] - t[SOLVED_JOINTS], axis=1)))
            for s, t in pairs
        )
        assert worst < 0.08, f"worst single joint error {worst:.3f} m"

    def test_beats_the_baseline_when_the_height_prior_is_wrong(self, scene):
        """The case that motivates the headset.

        The subject is 1.90 m and the tracker assumes 1.75 m. The geometric
        lifter infers depth from apparent bone size, so an 8% size error
        becomes an 8% depth error at three metres. The anchor knows better.
        """
        anchored = run(AnchoredLifter(), scene, make_context(scene, 1.75))
        geometric = run(GeometricLifter(), scene, make_context(scene, 1.75))

        anchored_error = median_joint_error(anchored)
        geometric_error = median_joint_error(geometric)
        assert anchored_error < geometric_error / 2.0, (
            f"anchored {anchored_error:.3f} m vs geometric {geometric_error:.3f} m"
        )

    def test_places_the_head_where_the_headset_says(self, scene):
        lifter = AnchoredLifter()
        pairs = run(lifter, scene, make_context(scene, 1.75))
        head_errors = [np.linalg.norm(s[HEAD] - t[HEAD]) for s, t in pairs]
        assert np.median(head_errors) < 0.02

    def test_absolute_depth_is_right_even_with_a_wrong_body_size(self, scene):
        """Shape may be off, but the subject must stand in the right place."""
        pairs = run(AnchoredLifter(), scene, make_context(scene, 1.75))
        hip_errors = [np.linalg.norm(s[HIP] - t[HIP]) for s, t in pairs]
        assert np.median(hip_errors) < 0.1

    def test_feet_stay_near_the_floor(self, scene):
        pairs = run(AnchoredLifter(), scene, make_context(scene, TRUE_HEIGHT))
        lowest = [min(s[LEFT_ANKLE, 1], s[RIGHT_ANKLE, 1]) for s, _ in pairs]
        assert np.median(lowest) < 0.2
        assert min(lowest) > -0.15


class TestResiduals:
    def test_wrist_residual_is_small_when_everything_is_right(self, scene):
        """The wrists are never pinned, so this is a genuine held-out check."""
        lifter = AnchoredLifter()
        run(lifter, scene, make_context(scene, TRUE_HEIGHT))
        assert lifter.wrist_residual_m is not None
        assert lifter.wrist_residual_m < 0.1

    def test_wrist_residual_grows_when_the_body_size_is_wrong(self, scene):
        """The residual has to be informative, or it is not worth reporting."""

        def residual(assumed_height: float) -> float:
            lifter = AnchoredLifter()
            values = []
            for frame in scene.frames[:20]:
                context = make_context(scene, assumed_height)
                context.vr = frame.vr
                lifter(frame.keypoints, context)
                if lifter.wrist_residual_m is not None:
                    values.append(lifter.wrist_residual_m)
            return float(np.median(values))

        assert residual(1.55) > residual(TRUE_HEIGHT)

    def test_reports_being_anchored(self, scene):
        lifter = AnchoredLifter()
        run(lifter, scene, make_context(scene, TRUE_HEIGHT), frames=5)
        assert lifter.diagnostics()["anchored"] is True


class TestGracefulDegradation:
    def test_falls_back_without_a_headset(self, scene):
        lifter = AnchoredLifter()
        context = make_context(scene, TRUE_HEIGHT)
        context.vr = None
        skeleton = lifter(scene.frames[0].keypoints, context)
        assert skeleton is not None
        assert lifter.anchored is False

    def test_falls_back_without_calibration(self, scene):
        """No extrinsics means no way to place the headset in camera space."""
        lifter = AnchoredLifter()
        context = make_context(scene, TRUE_HEIGHT, calibrated=False)
        context.vr = scene.frames[0].vr
        skeleton = lifter(scene.frames[0].keypoints, context)
        assert skeleton is not None
        assert lifter.anchored is False
        # The uncalibrated fallback still puts the subject on the floor.
        assert np.min(skeleton.xyz[:, 1]) == pytest.approx(0.0, abs=0.05)

    def test_stops_anchoring_when_the_model_tracks_someone_else(self, scene):
        """A bystander's keypoints must not drag the skeleton to the headset."""
        lifter = AnchoredLifter(max_head_residual_m=0.3)
        context = make_context(scene, TRUE_HEIGHT)

        context.vr = scene.frames[0].vr
        lifter(scene.frames[0].keypoints, context)
        assert lifter.anchored is True

        # Same headset pose, but the pose model is now looking two metres away.
        bystander = scene.frames[0].keypoints
        bystander.xy = bystander.xy + np.array([320.0, 0.0], dtype=np.float32)
        lifter(bystander, context)
        lifter(bystander, context)
        assert lifter.anchored is False
        assert lifter.rejected_anchors > 0

    def test_rejects_a_headset_behind_the_camera(self, scene):
        """Bad extrinsics can put the headset behind the lens; do not anchor."""
        from bodytracker.geom import look_at_extrinsics

        lifter = AnchoredLifter()
        context = make_context(scene, TRUE_HEIGHT)
        # A camera in the same place, turned to face away from the subject.
        rotation, translation = look_at_extrinsics(
            CAMERA_POSITION, CAMERA_POSITION + np.array([0.0, 0.0, 3.0])
        )
        context.rotation_cw, context.translation_cw = rotation, translation
        context.vr = scene.frames[0].vr
        lifter(scene.frames[0].keypoints, context)
        assert lifter.anchored is False
        assert lifter.rejected_anchors > 0

    def test_reset_clears_state(self, scene):
        lifter = AnchoredLifter()
        run(lifter, scene, make_context(scene, TRUE_HEIGHT), frames=5)
        lifter.reset()
        assert lifter.head_residual_m is None
        assert lifter.anchored is False


class TestOffsets:
    def test_uses_calibrated_offsets_when_given(self, scene):
        """Feeding back the calibrator's offsets should pin the head better.

        A wrong offset moves the anchor point, so the depth the head is pinned
        at is wrong by the offset's component along the view direction.
        """
        from bodytracker.calib.anchors import ANCHORS

        true_offsets = {a.name: a.local_offset.copy() for a in ANCHORS}
        true_offsets["head"] = true_offsets["head"] + np.array([0.0, 0.04, 0.09])
        shifted = build_scene(
            intrinsics=INTRINSICS,
            camera_position=CAMERA_POSITION,
            height_m=TRUE_HEIGHT,
            frames=20,
            offsets=true_offsets,
        )

        def head_error(lifter) -> float:
            pairs = run(lifter, shifted, make_context(shifted, TRUE_HEIGHT))
            return float(np.median([np.linalg.norm(s[HEAD] - t[HEAD]) for s, t in pairs]))

        assert head_error(AnchoredLifter(offsets=true_offsets)) < head_error(AnchoredLifter())


class TestAnchorPoints:
    def test_maps_devices_to_the_right_keypoints(self, scene):
        from bodytracker.lift import anchor_points_play

        frame = scene.frames[0]
        points = anchor_points_play(frame.vr)
        assert set(points) == {HEAD, LEFT_WRIST, RIGHT_WRIST}
        for joint, point in points.items():
            assert np.allclose(point, frame.play_xyz[joint], atol=1e-9)

    def test_skips_missing_devices(self, scene):
        from bodytracker.lift import anchor_points_play

        frame = scene.frames[0]
        frame.vr.left_hand = None
        assert LEFT_WRIST not in anchor_points_play(frame.vr)
