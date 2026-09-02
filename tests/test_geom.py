"""Coordinate space and rotation tests.

These are the highest-value tests in the repo. A sign error here does not
crash, it produces an avatar whose legs bend backwards, and it is nearly
impossible to diagnose from inside VRChat.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.geom import (
    camera_from_play,
    euler_zxy_degrees_from_matrix,
    intrinsics_from_fov,
    look_at_extrinsics,
    look_rotation,
    matrix_from_euler_zxy_degrees,
    matrix_from_quaternion,
    orthonormalize,
    play_from_camera,
    play_from_steamvr_matrix,
    project_points,
    quaternion_from_matrix,
    slerp,
    unity_position_from_play,
    unity_rotation_from_play,
)


def is_rotation(matrix: np.ndarray) -> bool:
    return bool(
        np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
        and np.isclose(np.linalg.det(matrix), 1.0, atol=1e-9)
    )


# ---------------------------------------------------------------------------
# Euler ZXY, the convention VRChat expects
# ---------------------------------------------------------------------------


class TestEulerZXY:
    @pytest.mark.parametrize(
        "angles",
        [
            (0.0, 0.0, 0.0),
            (30.0, 0.0, 0.0),
            (0.0, 45.0, 0.0),
            (0.0, 0.0, 60.0),
            (10.0, 20.0, 30.0),
            (-15.0, 170.0, -95.0),
            (89.0, -179.0, 45.0),
            (-89.0, 12.0, -170.0),
        ],
    )
    def test_roundtrip(self, angles):
        matrix = matrix_from_euler_zxy_degrees(*angles)
        assert is_rotation(matrix)
        recovered = euler_zxy_degrees_from_matrix(matrix)
        # Compare through the matrix: distinct Euler triples can name the same
        # rotation, so the angles themselves need not match bit for bit.
        assert np.allclose(matrix_from_euler_zxy_degrees(*recovered), matrix, atol=1e-9)

    def test_composition_order_is_y_x_z(self):
        """Unity applies Z, then X, then Y, which composes as Ry @ Rx @ Rz."""
        x, y, z = 25.0, -40.0, 70.0
        expected = (
            matrix_from_euler_zxy_degrees(0, y, 0)
            @ matrix_from_euler_zxy_degrees(x, 0, 0)
            @ matrix_from_euler_zxy_degrees(0, 0, z)
        )
        assert np.allclose(matrix_from_euler_zxy_degrees(x, y, z), expected, atol=1e-12)

    @pytest.mark.parametrize("pitch", [90.0, -90.0])
    def test_gimbal_lock_is_still_a_valid_rotation(self, pitch):
        """At x = +/-90 the decomposition degenerates; it must not produce NaN."""
        matrix = matrix_from_euler_zxy_degrees(pitch, 33.0, 21.0)
        recovered = euler_zxy_degrees_from_matrix(matrix)
        assert np.all(np.isfinite(recovered))
        assert np.allclose(matrix_from_euler_zxy_degrees(*recovered), matrix, atol=1e-7)

    def test_yaw_rotates_forward_toward_right(self):
        """A positive Y rotation must take +Z to +X, in both handedness rules."""
        rotated = matrix_from_euler_zxy_degrees(0, 90, 0) @ np.array([0.0, 0.0, 1.0])
        assert np.allclose(rotated, [1.0, 0.0, 0.0], atol=1e-12)

    def test_identity_is_zero(self):
        assert np.allclose(euler_zxy_degrees_from_matrix(np.eye(3)), [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Play <-> Unity handedness
# ---------------------------------------------------------------------------


class TestPlayToUnity:
    def test_position_flips_only_z(self):
        assert np.allclose(unity_position_from_play(np.array([1.0, 2.0, 3.0])), [1.0, 2.0, -3.0])

    def test_rotation_stays_a_rotation(self):
        """The reflection must be applied as a similarity transform.

        Negating a single row or column instead yields determinant -1, which is
        a mirror rather than a rotation, and shows up as an inside-out avatar.
        """
        rng = np.random.default_rng(0)
        for _ in range(50):
            play = orthonormalize(rng.normal(size=(3, 3)))
            unity = unity_rotation_from_play(play)
            assert is_rotation(unity)

    def test_is_self_inverse(self):
        rng = np.random.default_rng(1)
        play = orthonormalize(rng.normal(size=(3, 3)))
        assert np.allclose(unity_rotation_from_play(unity_rotation_from_play(play)), play)

    def test_conversion_commutes_with_rotating_a_point(self):
        """Converting then rotating must equal rotating then converting."""
        rng = np.random.default_rng(2)
        rot_play = orthonormalize(rng.normal(size=(3, 3)))
        point_play = rng.normal(size=3)

        via_play = unity_position_from_play(rot_play @ point_play)
        via_unity = unity_rotation_from_play(rot_play) @ unity_position_from_play(point_play)
        assert np.allclose(via_play, via_unity, atol=1e-12)

    def test_play_yaw_maps_to_opposite_unity_yaw(self):
        """Handedness flip reverses the sense of rotation about the shared up axis."""
        play_yaw = matrix_from_euler_zxy_degrees(0, 30, 0)
        unity = unity_rotation_from_play(play_yaw)
        assert np.allclose(euler_zxy_degrees_from_matrix(unity), [0.0, -30.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Camera <-> play
# ---------------------------------------------------------------------------


class TestCameraPlay:
    def test_roundtrip_single_point(self):
        rot, trans = look_at_extrinsics(np.array([2.0, 1.5, 2.0]), np.array([0.0, 1.0, 0.0]))
        point = np.array([0.3, 1.2, -0.4])
        assert np.allclose(play_from_camera(camera_from_play(point, rot, trans), rot, trans), point)

    def test_roundtrip_batch(self):
        rng = np.random.default_rng(3)
        rot, trans = look_at_extrinsics(np.array([0.0, 1.4, 3.0]), np.array([0.0, 1.0, 0.0]))
        points = rng.normal(size=(20, 3))
        assert np.allclose(
            play_from_camera(camera_from_play(points, rot, trans), rot, trans), points
        )

    def test_preserves_shape(self):
        rot, trans = look_at_extrinsics(np.array([0.0, 1.4, 3.0]), np.zeros(3))
        assert camera_from_play(np.zeros(3), rot, trans).shape == (3,)
        assert camera_from_play(np.zeros((5, 3)), rot, trans).shape == (5, 3)

    def test_target_lands_on_optical_axis(self):
        """A camera aimed at a point should see it dead centre."""
        target = np.array([0.0, 1.0, 0.0])
        rot, trans = look_at_extrinsics(np.array([0.0, 1.4, 3.0]), target)
        in_camera = camera_from_play(target, rot, trans)
        assert in_camera[2] > 0, "target must be in front of the camera"
        assert np.allclose(in_camera[:2], [0.0, 0.0], atol=1e-9)

    def test_camera_y_points_down(self):
        """OpenCV camera space is +y down, so a higher point has smaller y."""
        rot, trans = look_at_extrinsics(np.array([0.0, 1.4, 3.0]), np.array([0.0, 1.0, 0.0]))
        high = camera_from_play(np.array([0.0, 1.8, 0.0]), rot, trans)
        low = camera_from_play(np.array([0.0, 0.2, 0.0]), rot, trans)
        assert high[1] < low[1]

    def test_extrinsics_are_a_rigid_transform(self):
        rot, _ = look_at_extrinsics(np.array([1.0, 1.4, 2.0]), np.array([0.0, 1.0, 0.0]))
        assert is_rotation(rot)


class TestProjection:
    def test_centre_of_projection(self):
        mat = intrinsics_from_fov(1280, 720, 62.0)
        assert np.allclose(project_points(np.array([0.0, 0.0, 2.0]), mat), [640.0, 360.0])

    def test_fov_edge_lands_on_image_edge(self):
        """A point at exactly half the horizontal FOV projects to the frame edge."""
        fov = 62.0
        mat = intrinsics_from_fov(1280, 720, fov)
        depth = 2.0
        half_width = depth * np.tan(np.radians(fov) / 2.0)
        pixel = project_points(np.array([half_width, 0.0, depth]), mat)
        assert np.isclose(pixel[0], 1280.0, atol=1e-6)

    def test_batch_shape(self):
        mat = intrinsics_from_fov(1280, 720, 62.0)
        assert project_points(np.ones((7, 3)), mat).shape == (7, 2)


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------


class TestRotationUtilities:
    def test_orthonormalize_fixes_drift(self):
        rng = np.random.default_rng(4)
        clean = orthonormalize(rng.normal(size=(3, 3)))
        drifted = clean + rng.normal(scale=1e-3, size=(3, 3))
        assert is_rotation(orthonormalize(drifted))

    def test_orthonormalize_rejects_reflections(self):
        reflection = np.diag([1.0, 1.0, -1.0])
        assert np.isclose(np.linalg.det(orthonormalize(reflection)), 1.0)

    def test_look_rotation_aligns_z_with_forward(self):
        forward = np.array([1.0, 0.5, -2.0])
        rot = look_rotation(forward)
        assert is_rotation(rot)
        assert np.allclose(rot @ np.array([0.0, 0.0, 1.0]), forward / np.linalg.norm(forward))

    def test_look_rotation_handles_parallel_up(self):
        rot = look_rotation(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        assert is_rotation(rot)

    def test_look_rotation_handles_degenerate_forward(self):
        assert np.allclose(look_rotation(np.zeros(3)), np.eye(3))

    def test_quaternion_roundtrip(self):
        rng = np.random.default_rng(5)
        for _ in range(100):
            rot = orthonormalize(rng.normal(size=(3, 3)))
            assert np.allclose(matrix_from_quaternion(quaternion_from_matrix(rot)), rot, atol=1e-9)

    @pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    def test_quaternion_stable_at_180_degrees(self, axis):
        """The trace branch is degenerate at 180 degrees; Shepperd's method must hold."""
        angle = np.pi
        ax = np.array(axis, dtype=np.float64)
        skew = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        rot = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        assert np.allclose(matrix_from_quaternion(quaternion_from_matrix(rot)), rot, atol=1e-9)

    def test_slerp_endpoints(self):
        a = matrix_from_euler_zxy_degrees(0, 0, 0)
        b = matrix_from_euler_zxy_degrees(0, 80, 0)
        assert np.allclose(slerp(a, b, 0.0), a, atol=1e-9)
        assert np.allclose(slerp(a, b, 1.0), b, atol=1e-9)

    def test_slerp_midpoint_is_halfway(self):
        a = matrix_from_euler_zxy_degrees(0, 0, 0)
        b = matrix_from_euler_zxy_degrees(0, 80, 0)
        mid = euler_zxy_degrees_from_matrix(slerp(a, b, 0.5))
        assert np.allclose(mid, [0.0, 40.0, 0.0], atol=1e-9)

    def test_slerp_takes_short_way_around(self):
        """Interpolating 350 degrees should sweep 10 degrees, not 350."""
        a = matrix_from_euler_zxy_degrees(0, 0, 0)
        b = matrix_from_euler_zxy_degrees(0, 350, 0)
        mid = euler_zxy_degrees_from_matrix(slerp(a, b, 0.5))
        assert np.allclose(mid, [0.0, -5.0, 0.0], atol=1e-9)

    def test_slerp_output_is_a_rotation(self):
        rng = np.random.default_rng(6)
        a = orthonormalize(rng.normal(size=(3, 3)))
        b = orthonormalize(rng.normal(size=(3, 3)))
        for t in np.linspace(0, 1, 11):
            assert is_rotation(slerp(a, b, float(t)))


class TestSteamVRMatrix:
    def test_splits_position_and_rotation(self):
        mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, 1.7],
                [0.0, 0.0, 1.0, -0.3],
            ]
        )
        position, rotation = play_from_steamvr_matrix(mat)
        assert np.allclose(position, [0.5, 1.7, -0.3])
        assert np.allclose(rotation, np.eye(3))

    def test_accepts_flat_input(self):
        flat = np.arange(12, dtype=np.float64)
        position, rotation = play_from_steamvr_matrix(flat)
        assert np.allclose(position, [3.0, 7.0, 11.0])
        assert rotation.shape == (3, 3)
