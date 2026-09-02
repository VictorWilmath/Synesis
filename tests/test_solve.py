"""Tests for orientation derivation and physical constraints."""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.geom import euler_zxy_degrees_from_matrix, unity_rotation_from_play
from bodytracker.skeleton import (
    LEFT_ANKLE,
    LEFT_HEEL,
    LEFT_KNEE,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    TrackerRole,
)
from bodytracker.solve import (
    build_targets,
    chest_rotation,
    clamp_to_floor,
    derive_rotations,
    floor_penetration,
    foot_rotation,
    knee_rotation,
    pelvis_rotation,
)
from bodytracker.tools.synthetic import reference_skeleton
from bodytracker.types import Skeleton3D

HEIGHT = 1.75


def yaw_of(rotation: np.ndarray) -> float:
    """Unity-space yaw in degrees, which is what VRChat receives."""
    return float(euler_zxy_degrees_from_matrix(unity_rotation_from_play(rotation))[1])


def is_rotation(matrix: np.ndarray) -> bool:
    return bool(
        np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8)
        and np.isclose(np.linalg.det(matrix), 1.0, atol=1e-8)
    )


def yaw_skeleton(xyz: np.ndarray, degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
    return xyz @ rot.T


class TestPelvisRotation:
    def test_forward_facing_subject_is_identity_in_unity(self):
        """The reference skeleton faces SteamVR-forward, so yaw must be zero."""
        rot = pelvis_rotation(reference_skeleton(HEIGHT))
        assert rot is not None and is_rotation(rot)
        assert yaw_of(rot) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("degrees", [-135.0, -90.0, -30.0, 30.0, 90.0, 135.0])
    def test_tracks_body_yaw(self, degrees):
        """Rotating the body must rotate the hip tracker by the same amount.

        The play-to-Unity flip inverts the sense, which is exactly the sort of
        error that leaves an avatar facing backwards.
        """
        rotated = yaw_skeleton(reference_skeleton(HEIGHT), degrees)
        assert yaw_of(pelvis_rotation(rotated)) == pytest.approx(-degrees, abs=1e-6)

    def test_degenerate_skeleton_returns_none(self):
        assert pelvis_rotation(np.zeros((NUM_KEYPOINTS, 3))) is None


class TestChestRotation:
    def test_forward_facing(self):
        assert yaw_of(chest_rotation(reference_skeleton(HEIGHT))) == pytest.approx(0.0, abs=1e-6)

    def test_captures_torso_twist_independently_of_hips(self):
        """Shoulders turned while hips stay put must show up as chest yaw only."""
        from bodytracker.skeleton import LEFT_SHOULDER, NECK, RIGHT_SHOULDER

        xyz = reference_skeleton(HEIGHT)
        half_span = (xyz[RIGHT_SHOULDER, 0] - xyz[LEFT_SHOULDER, 0]) / 2.0
        angle = np.radians(40.0)
        centre = xyz[NECK].copy()
        xyz[LEFT_SHOULDER] = centre + [
            -half_span * np.cos(angle),
            0.0,
            half_span * np.sin(angle),
        ]
        xyz[RIGHT_SHOULDER] = centre + [
            half_span * np.cos(angle),
            0.0,
            -half_span * np.sin(angle),
        ]

        assert abs(yaw_of(chest_rotation(xyz))) > 30.0
        assert yaw_of(pelvis_rotation(xyz)) == pytest.approx(0.0, abs=1e-6)


class TestFootRotation:
    def test_points_forward_for_a_standing_subject(self):
        rot = foot_rotation(reference_skeleton(HEIGHT), left=True)
        assert rot is not None and is_rotation(rot)
        assert yaw_of(rot) == pytest.approx(0.0, abs=5.0)

    def test_both_feet_agree_when_standing(self):
        xyz = reference_skeleton(HEIGHT)
        assert yaw_of(foot_rotation(xyz, left=True)) == pytest.approx(
            yaw_of(foot_rotation(xyz, left=False)), abs=1e-6
        )

    def test_tracks_foot_yaw(self):
        """Turning a foot outward must show up as tracker yaw."""
        from bodytracker.skeleton import LEFT_BIG_TOE, LEFT_SMALL_TOE

        xyz = reference_skeleton(HEIGHT)
        pivot = xyz[LEFT_HEEL].copy()
        angle = np.radians(35.0)
        rot = np.array(
            [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        )
        for joint in (LEFT_BIG_TOE, LEFT_SMALL_TOE, LEFT_ANKLE):
            xyz[joint] = pivot + rot @ (xyz[joint] - pivot)

        assert abs(yaw_of(foot_rotation(xyz, left=True))) > 20.0

    def test_stays_upright_despite_noisy_toe_keypoints(self):
        """Toe keypoints are small and noisy; the foot must not flip over."""
        from bodytracker.skeleton import LEFT_BIG_TOE, LEFT_SMALL_TOE

        rng = np.random.default_rng(0)
        for _ in range(100):
            xyz = reference_skeleton(HEIGHT)
            xyz[LEFT_BIG_TOE] += rng.normal(scale=0.02, size=3)
            xyz[LEFT_SMALL_TOE] += rng.normal(scale=0.02, size=3)
            rot = foot_rotation(xyz, left=True)
            # Local up must remain broadly world-up.
            assert float(rot[:, 1] @ np.array([0.0, 1.0, 0.0])) > 0.5


class TestKneeRotation:
    def test_straight_leg_is_degenerate(self):
        """A straight leg carries no information about the direction of bend."""
        assert knee_rotation(reference_skeleton(HEIGHT), left=True) is None

    def test_bent_knee_points_the_way_it_bends(self):
        xyz = reference_skeleton(HEIGHT)
        # Push the knee forward (-z) to bend the leg.
        xyz[LEFT_KNEE, 2] -= 0.25
        rot = knee_rotation(xyz, left=True)
        assert rot is not None and is_rotation(rot)
        forward = rot[:, 2]
        # Local +z is forward in the SteamVR convention, so the bend direction
        # is the negated third column.
        assert float(-forward @ np.array([0.0, 0.0, -1.0])) > 0.5


class TestDeriveRotations:
    def test_covers_every_role(self):
        rotations = derive_rotations(reference_skeleton(HEIGHT))
        assert set(rotations) == set(TrackerRole)

    def test_all_outputs_are_valid_rotations(self):
        for rot in derive_rotations(reference_skeleton(HEIGHT)).values():
            assert is_rotation(rot)

    def test_falls_back_to_the_pelvis_for_degenerate_joints(self):
        """Straight limbs give no knee or elbow orientation, so reuse the hips."""
        xyz = reference_skeleton(HEIGHT)
        rotations = derive_rotations(xyz)
        assert np.allclose(rotations[TrackerRole.LEFT_KNEE], rotations[TrackerRole.HIP])

    def test_never_produces_nan_on_garbage_input(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            xyz = rng.normal(scale=0.5, size=(NUM_KEYPOINTS, 3))
            for rot in derive_rotations(xyz).values():
                assert np.all(np.isfinite(rot))

    def test_survives_an_all_zero_skeleton(self):
        for rot in derive_rotations(np.zeros((NUM_KEYPOINTS, 3))).values():
            assert np.all(np.isfinite(rot)) and is_rotation(rot)


class TestFloorConstraints:
    def test_no_penetration_for_a_standing_skeleton(self):
        assert floor_penetration(reference_skeleton(HEIGHT)) == pytest.approx(0.0, abs=1e-9)

    def test_measures_how_far_below_the_floor(self):
        xyz = reference_skeleton(HEIGHT)
        xyz[:, 1] -= 0.2
        assert floor_penetration(xyz) == pytest.approx(0.2, abs=1e-6)

    def test_lifts_a_sunken_body_whole(self):
        """Clamping each foot separately would compress the legs instead."""
        xyz = reference_skeleton(HEIGHT)
        sunk = xyz.copy()
        sunk[:, 1] -= 0.3
        fixed = clamp_to_floor(sunk, epsilon=0.03)
        assert floor_penetration(fixed) == pytest.approx(0.0, abs=1e-9)
        # Shape preserved: the body moved rigidly.
        offsets = fixed - sunk
        assert np.allclose(offsets, offsets[0], atol=1e-9)

    def test_clamps_a_single_dipping_foot(self):
        xyz = reference_skeleton(HEIGHT)
        xyz[LEFT_ANKLE, 1] = -0.01
        fixed = clamp_to_floor(xyz, epsilon=0.03)
        assert fixed[LEFT_ANKLE, 1] >= 0.0
        assert fixed[RIGHT_ANKLE, 1] == pytest.approx(xyz[RIGHT_ANKLE, 1])

    def test_leaves_an_airborne_subject_alone(self):
        xyz = reference_skeleton(HEIGHT)
        xyz[:, 1] += 0.4
        assert np.allclose(clamp_to_floor(xyz), xyz)


class TestBuildTargets:
    def test_one_target_per_requested_role(self):
        skeleton = Skeleton3D(
            reference_skeleton(HEIGHT).astype(np.float32),
            np.ones(NUM_KEYPOINTS, dtype=np.float32),
        )
        roles = [TrackerRole.HIP, TrackerRole.LEFT_FOOT, TrackerRole.RIGHT_FOOT]
        targets = build_targets(skeleton, roles)
        assert [t.role for t in targets] == roles

    def test_anchors_positions_to_the_right_joints(self):
        xyz = reference_skeleton(HEIGHT)
        skeleton = Skeleton3D(xyz.astype(np.float32), np.ones(NUM_KEYPOINTS, dtype=np.float32))
        targets = {t.role: t for t in build_targets(skeleton, list(TrackerRole))}
        assert np.allclose(targets[TrackerRole.LEFT_FOOT].position, xyz[LEFT_ANKLE], atol=1e-5)

    def test_marks_low_confidence_targets_invalid_but_still_returns_them(self):
        scores = np.ones(NUM_KEYPOINTS, dtype=np.float32)
        scores[LEFT_ANKLE] = 0.01
        skeleton = Skeleton3D(reference_skeleton(HEIGHT).astype(np.float32), scores)
        targets = {t.role: t for t in build_targets(skeleton, list(TrackerRole))}
        assert targets[TrackerRole.LEFT_FOOT].valid is False
        assert targets[TrackerRole.HIP].valid is True
