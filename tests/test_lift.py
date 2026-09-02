"""Tests for the geometric lifting math.

The key test is the round trip: take a skeleton with known bone lengths,
project it through a pinhole camera, lift it back, and check we recover what we
started with. Anything the lifter gets wrong there it will get wrong on real
keypoints too, only with noise on top hiding it.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.geom import camera_from_play, look_at_extrinsics, project_points
from bodytracker.geom.spaces import intrinsics_from_fov
from bodytracker.lift import (
    bone_length_lookup,
    bone_lengths_from_height,
    enforce_bone_lengths,
    estimate_root_depth,
    lift_by_bone_lengths,
    measure_bone_lengths,
    propagate_depths,
    solve_child_depth,
)
from bodytracker.lift.kinematics import backproject_rays, body_forward, resolve_foot_depths
from bodytracker.skeleton import (
    BONES,
    DEPTH_COPY_JOINTS,
    HEAD,
    HIP,
    KINEMATIC_PARENT,
    LATERAL_PAIRS,
    LEFT_ANKLE,
    LEFT_BIG_TOE,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_SMALL_TOE,
    NUM_KEYPOINTS,
    RIGHT_BIG_TOE,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_SMALL_TOE,
    ROOT,
    TOPOLOGICAL_ORDER,
)
from bodytracker.tools.synthetic import reference_skeleton

HEIGHT = 1.75
INTRINSICS = intrinsics_from_fov(1280, 720, 62.0)


@pytest.fixture
def scene():
    """A reference skeleton seen by a camera 3 m away at chest height."""
    play = reference_skeleton(HEIGHT)
    rot, trans = look_at_extrinsics(np.array([0.0, 1.2, 3.0]), np.array([0.0, 0.9, 0.0]))
    camera = camera_from_play(play, rot, trans)
    pixels = project_points(camera, INTRINSICS)
    return play, camera, pixels


# ---------------------------------------------------------------------------
# Kinematic tree
# ---------------------------------------------------------------------------


class TestKinematicTree:
    def test_topological_order_covers_every_joint(self):
        assert sorted(TOPOLOGICAL_ORDER) == list(range(NUM_KEYPOINTS))

    def test_parents_precede_children(self):
        position = {joint: i for i, joint in enumerate(TOPOLOGICAL_ORDER)}
        for joint in range(NUM_KEYPOINTS):
            parent = KINEMATIC_PARENT[joint]
            if parent is not None:
                assert position[parent] < position[joint]

    def test_single_root(self):
        roots = [j for j in range(NUM_KEYPOINTS) if KINEMATIC_PARENT[j] is None]
        assert roots == [ROOT] == [HIP]

    def test_every_bone_maps_to_a_parent_link(self):
        """The bone table and the kinematic tree must agree."""
        lookup = bone_length_lookup(bone_lengths_from_height(HEIGHT))
        for a, b in BONES:
            assert KINEMATIC_PARENT[a] == b or KINEMATIC_PARENT[b] == a
        child = max(BONES, key=lambda ab: ab[1])[1]
        assert lookup[child] > 0


class TestReferenceSkeleton:
    def test_bone_lengths_match_the_prior(self):
        """The fixture must be exact, or round-trip failures are ambiguous."""
        xyz = reference_skeleton(HEIGHT)
        expected = bone_lengths_from_height(HEIGHT)
        measured = measure_bone_lengths(xyz)
        for bone, target in expected.items():
            assert measured[bone] == pytest.approx(target, abs=1e-9), bone

    def test_feet_rest_on_the_floor(self):
        xyz = reference_skeleton(HEIGHT)
        assert np.min(xyz[:, 1]) == pytest.approx(0.0, abs=0.02)

    def test_scales_with_height(self):
        small = reference_skeleton(1.5)
        large = reference_skeleton(2.0)
        assert np.max(large[:, 1]) > np.max(small[:, 1])


# ---------------------------------------------------------------------------
# Depth solving
# ---------------------------------------------------------------------------


class TestSolveChildDepth:
    def test_recovers_a_known_depth(self):
        parent_ray = np.array([0.0, 0.0, 1.0])
        child_ray = np.array([0.1, 0.0, 1.0])
        parent_depth = 3.0
        child_depth = 3.2
        length = float(np.linalg.norm(child_depth * child_ray - parent_depth * parent_ray))

        solved = solve_child_depth(parent_depth, parent_ray, child_ray, length)
        assert solved == pytest.approx(child_depth, abs=1e-6)

    def test_picks_the_root_nearest_the_parent(self):
        """Both roots are geometrically valid; the near one is the prior."""
        parent_ray = np.array([0.0, 0.0, 1.0])
        child_ray = np.array([0.05, 0.0, 1.0])
        solved = solve_child_depth(3.0, parent_ray, child_ray, 0.4)
        assert abs(solved - 3.0) < 0.4

    def test_handles_impossible_geometry(self):
        """A bone too short to span the observed 2D gap must not produce NaN."""
        parent_ray = np.array([0.0, 0.0, 1.0])
        child_ray = np.array([2.0, 0.0, 1.0])
        solved = solve_child_depth(3.0, parent_ray, child_ray, 0.01)
        assert np.isfinite(solved)
        assert solved > 0

    def test_output_is_always_in_front_of_the_camera(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            parent_ray = np.array([rng.normal(0, 0.3), rng.normal(0, 0.3), 1.0])
            child_ray = np.array([rng.normal(0, 0.3), rng.normal(0, 0.3), 1.0])
            solved = solve_child_depth(
                float(rng.uniform(1.0, 5.0)), parent_ray, child_ray, float(rng.uniform(0.05, 0.6))
            )
            assert np.isfinite(solved) and solved > 0


class TestRootDepthEstimate:
    def test_close_to_truth_for_a_frontal_pose(self, scene):
        _, camera, pixels = scene
        rays = backproject_rays(pixels, INTRINSICS)
        estimate = estimate_root_depth(
            rays, bone_lengths_from_height(HEIGHT), np.ones(NUM_KEYPOINTS), 0.3
        )
        assert estimate == pytest.approx(camera[HIP, 2], rel=0.15)

    def test_returns_none_without_enough_confident_joints(self):
        rays = np.tile([0.0, 0.0, 1.0], (NUM_KEYPOINTS, 1)).astype(float)
        scores = np.zeros(NUM_KEYPOINTS)
        assert estimate_root_depth(rays, bone_lengths_from_height(HEIGHT), scores, 0.3) is None


class TestRoundTrip:
    def test_recovers_camera_space_given_true_root_depth(self, scene):
        """With the pelvis pinned, the tree walk should reproduce the skeleton.

        This is the situation the HMD anchor creates in Phase 2.

        The foot extremities are excluded: the ankle-to-toe and ankle-to-heel
        bones point almost straight down the optical axis in this camera
        placement, which is exactly the configuration where the two depth roots
        are indistinguishable. The next test shows a depth prior resolving it.
        """
        _, camera, pixels = scene
        lifted = lift_by_bone_lengths(
            pixels,
            np.ones(NUM_KEYPOINTS),
            INTRINSICS,
            bone_lengths_from_height(HEIGHT),
            root_depth=float(camera[HIP, 2]),
        )
        assert lifted is not None
        error = np.linalg.norm(lifted - camera, axis=1)
        assert np.median(error) < 0.02, f"median joint error {np.median(error):.4f} m"

        # Everything but the depth-copied face and the short foot bones that
        # lie along the view axis here.
        ambiguous = {
            LEFT_BIG_TOE,
            RIGHT_BIG_TOE,
            LEFT_SMALL_TOE,
            RIGHT_SMALL_TOE,
            LEFT_HEEL,
            RIGHT_HEEL,
        }
        solved = [
            i for i in range(NUM_KEYPOINTS) if i not in ambiguous and i not in DEPTH_COPY_JOINTS
        ]
        assert np.max(error[solved]) < 0.03

    def test_depth_prior_resolves_the_foreshortening_ambiguity(self, scene):
        """Seeded with the true depths, every joint should come back exactly.

        This is what the runtime does with the previous frame, and why limbs
        pointing at the camera stay put instead of flipping.
        """
        _, camera, pixels = scene
        lifted = lift_by_bone_lengths(
            pixels,
            np.ones(NUM_KEYPOINTS),
            INTRINSICS,
            bone_lengths_from_height(HEIGHT),
            root_depth=float(camera[HIP, 2]),
            depth_prior=camera[:, 2].copy(),
        )
        error = np.linalg.norm(lifted - camera, axis=1)
        solved = [i for i in range(NUM_KEYPOINTS) if i not in DEPTH_COPY_JOINTS]
        assert np.max(error[solved]) < 0.02, f"worst joint error {np.max(error[solved]):.4f} m"

    def test_face_joints_inherit_head_depth(self, scene):
        """Face segments are centimetres long, so solving them amplifies noise.

        They are deliberately assigned the head's depth instead, and nothing
        downstream consumes face depth.
        """
        _, camera, pixels = scene
        lifted = lift_by_bone_lengths(
            pixels,
            np.ones(NUM_KEYPOINTS),
            INTRINSICS,
            bone_lengths_from_height(HEIGHT),
            root_depth=float(camera[HIP, 2]),
        )
        for joint in DEPTH_COPY_JOINTS:
            assert lifted[joint, 2] == pytest.approx(lifted[HEAD, 2], abs=1e-9)

    def test_depth_prior_survives_the_subject_moving(self, scene):
        """The prior is offset by root motion, so walking must not break it."""
        _, camera, _ = scene
        play = reference_skeleton(HEIGHT)
        # Same pose, half a metre closer to the camera.
        rot, trans = look_at_extrinsics(np.array([0.0, 1.2, 2.5]), np.array([0.0, 0.9, 0.0]))
        moved = camera_from_play(play, rot, trans)
        pixels = project_points(moved, INTRINSICS)

        lifted = lift_by_bone_lengths(
            pixels,
            np.ones(NUM_KEYPOINTS),
            INTRINSICS,
            bone_lengths_from_height(HEIGHT),
            root_depth=float(moved[HIP, 2]),
            depth_prior=camera[:, 2].copy(),
        )
        error = np.linalg.norm(lifted - moved, axis=1)
        solved = [i for i in range(NUM_KEYPOINTS) if i not in DEPTH_COPY_JOINTS]
        assert np.max(error[solved]) < 0.02

    def test_reprojects_onto_the_original_pixels(self, scene):
        """Whatever the depth error, the solution must stay on the camera rays."""
        _, camera, pixels = scene
        lifted = lift_by_bone_lengths(
            pixels,
            np.ones(NUM_KEYPOINTS),
            INTRINSICS,
            bone_lengths_from_height(HEIGHT),
            root_depth=float(camera[HIP, 2]),
        )
        reprojected = project_points(lifted, INTRINSICS)
        assert np.max(np.abs(reprojected - pixels)) < 1.0

    def test_unpinned_lift_is_plausible(self, scene):
        """Without a root depth the scale is a guess, but shape should survive."""
        _, camera, pixels = scene
        lifted = lift_by_bone_lengths(
            pixels, np.ones(NUM_KEYPOINTS), INTRINSICS, bone_lengths_from_height(HEIGHT)
        )
        assert lifted is not None
        # Compare shape after removing the root offset.
        centred_truth = camera - camera[HIP]
        centred_lift = lifted - lifted[HIP]
        assert np.median(np.linalg.norm(centred_lift - centred_truth, axis=1)) < 0.1

    def test_propagated_depths_are_all_positive(self, scene):
        _, camera, pixels = scene
        rays = backproject_rays(pixels, INTRINSICS)
        depths = propagate_depths(rays, float(camera[HIP, 2]), bone_lengths_from_height(HEIGHT))
        assert np.all(depths > 0)
        assert np.all(np.isfinite(depths))

    @pytest.mark.parametrize("root", [HIP, HEAD, LEFT_ANKLE])
    def test_any_joint_can_be_the_anchor(self, scene, root):
        """Re-rooting is what lets the HMD pin the head rather than the pelvis."""
        _, camera, pixels = scene
        lifted = lift_by_bone_lengths(
            pixels,
            np.ones(NUM_KEYPOINTS),
            INTRINSICS,
            bone_lengths_from_height(HEIGHT),
            root_depth=float(camera[root, 2]),
            depth_prior=camera[:, 2].copy(),
            root_joint=root,
        )
        error = np.linalg.norm(lifted - camera, axis=1)
        solved = [i for i in range(NUM_KEYPOINTS) if i not in DEPTH_COPY_JOINTS]
        assert np.max(error[solved]) < 0.02
        assert lifted[root, 2] == pytest.approx(camera[root, 2], abs=1e-9)


class TestDisambiguation:
    """The depth quadratic has two roots; these check we pick the right one.

    Every case here is one where the plain "keep the child near the parent's
    depth" rule picks wrong, because the ambiguity is not resolved by depth
    proximity but by knowing something about how bodies are arranged.
    """

    @staticmethod
    def _turned(yaw_deg: float, camera_height: float = 1.2):
        play = reference_skeleton(HEIGHT)
        angle = np.radians(yaw_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        yaw = np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])
        play = play @ yaw.T + np.array([0.0, 0.0, 0.3])

        rot, trans = look_at_extrinsics(
            np.array([0.0, camera_height, 3.0]), np.array([0.0, 0.9, 0.0])
        )
        camera = camera_from_play(play, rot, trans)
        pixels = project_points(camera, INTRINSICS)
        up_camera = rot @ np.array([0.0, 1.0, 0.0])
        return camera, pixels, up_camera

    @pytest.mark.parametrize("camera_height", [0.8, 1.1, 1.4, 1.7, 2.0])
    def test_gravity_always_recovers_an_upright_spine(self, camera_height):
        camera, pixels, up_camera = self._turned(0.0, camera_height=camera_height)
        rays = backproject_rays(pixels, INTRINSICS)
        depths = propagate_depths(
            rays,
            float(camera[HEAD, 2]),
            bone_lengths_from_height(HEIGHT),
            root_joint=HEAD,
            up_camera=up_camera,
        )
        assert abs(depths[HIP] - camera[HIP, 2]) < 0.001

    def test_depth_proximity_alone_would_get_some_of_those_wrong(self):
        """Establishes that the gravity prior is earning its place.

        A camera that looks even slightly downward sees a vertical spine as
        sloping away from it, so the most fronto-parallel solution is the
        mirrored one and the pelvis lands several centimetres out.
        """
        lengths = bone_lengths_from_height(HEIGHT)
        errors = []
        for camera_height in (0.8, 1.1, 1.4, 1.7, 2.0):
            camera, pixels, _ = self._turned(0.0, camera_height=camera_height)
            rays = backproject_rays(pixels, INTRINSICS)
            depths = propagate_depths(rays, float(camera[HEAD, 2]), lengths, root_joint=HEAD)
            errors.append(abs(depths[HIP] - camera[HIP, 2]))

        assert max(errors) > 0.02

    @pytest.mark.parametrize("lean_deg", [30.0, 40.0])
    def test_gravity_does_not_straighten_a_bent_subject(self, lean_deg):
        """Gravity must abstain when the body genuinely is not upright.

        Both candidates lean then, and picking the less tilted one would stand
        the subject up. Better to defer and let the temporal prior decide.
        """
        play = reference_skeleton(HEIGHT)
        angle = np.radians(lean_deg)
        pitch = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        play = play @ pitch.T + np.array([0.0, 0.2, 0.3])

        rot, trans = look_at_extrinsics(np.array([0.0, 1.2, 3.0]), np.array([0.0, 0.9, 0.0]))
        camera = camera_from_play(play, rot, trans)
        rays = backproject_rays(project_points(camera, INTRINSICS), INTRINSICS)

        depths = propagate_depths(
            rays,
            float(camera[HEAD, 2]),
            bone_lengths_from_height(HEIGHT),
            depth_prior=camera[:, 2].copy(),
            root_joint=HEAD,
            up_camera=rot @ np.array([0.0, 1.0, 0.0]),
        )
        assert abs(depths[HIP] - camera[HIP, 2]) < 0.01

    def test_the_temporal_prior_outranks_gravity(self):
        """Memory of the actual pose beats an assumption about the usual one."""
        play = reference_skeleton(HEIGHT)
        angle = np.radians(35.0)
        pitch = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        play = play @ pitch.T + np.array([0.0, 0.2, 0.3])
        rot, trans = look_at_extrinsics(np.array([0.0, 1.2, 3.0]), np.array([0.0, 0.9, 0.0]))
        camera = camera_from_play(play, rot, trans)
        rays = backproject_rays(project_points(camera, INTRINSICS), INTRINSICS)
        lengths = bone_lengths_from_height(HEIGHT)
        up_camera = rot @ np.array([0.0, 1.0, 0.0])

        informed = propagate_depths(
            rays,
            float(camera[HEAD, 2]),
            lengths,
            depth_prior=camera[:, 2].copy(),
            root_joint=HEAD,
            up_camera=up_camera,
        )
        for joint in range(NUM_KEYPOINTS):
            if joint in DEPTH_COPY_JOINTS:
                continue
            assert abs(informed[joint] - camera[joint, 2]) < 0.01, joint

    @pytest.mark.parametrize("yaw", [-50.0, -25.0, 25.0, 50.0])
    def test_shoulders_stay_on_opposite_sides_of_the_neck(self, yaw):
        """Solved one at a time, both shoulders can land on the same side."""
        camera, pixels, up_camera = self._turned(yaw)
        rays = backproject_rays(pixels, INTRINSICS)
        lengths = bone_lengths_from_height(HEIGHT)

        depths = propagate_depths(
            rays, float(camera[HEAD, 2]), lengths, root_joint=HEAD, up_camera=up_camera
        )
        xyz = rays * depths[:, None]

        for parent, left, right in LATERAL_PAIRS:
            to_left = xyz[left] - xyz[parent]
            to_right = xyz[right] - xyz[parent]
            cosine = np.dot(to_left, to_right) / (
                np.linalg.norm(to_left) * np.linalg.norm(to_right)
            )
            assert cosine < -0.8, f"{parent} pair is not straddled at yaw {yaw}"

    @pytest.mark.parametrize("yaw", [-45.0, 45.0])
    def test_the_facing_hint_resolves_the_mirror_ambiguity(self, yaw):
        """Turned left and turned right project almost identically.

        Only the headset can say which it is, so without the hint the solve is
        allowed to be wrong and with it must not be.
        """
        camera, pixels, up_camera = self._turned(yaw)
        rays = backproject_rays(pixels, INTRINSICS)
        lengths = bone_lengths_from_height(HEIGHT)

        rot, _ = look_at_extrinsics(np.array([0.0, 1.2, 3.0]), np.array([0.0, 0.9, 0.0]))
        angle = np.radians(yaw)
        facing_play = np.array([-np.sin(angle), 0.0, -np.cos(angle)])

        depths = propagate_depths(
            rays,
            float(camera[HEAD, 2]),
            lengths,
            root_joint=HEAD,
            up_camera=up_camera,
            facing_camera=rot @ facing_play,
        )
        for joint in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
            assert abs(depths[joint] - camera[joint, 2]) < 0.01, joint

    def test_feet_point_the_way_the_body_faces(self):
        """Toe bones point down the optical axis, so no depth rule can help."""
        camera, pixels, up_camera = self._turned(0.0)
        rays = backproject_rays(pixels, INTRINSICS)
        lengths = bone_lengths_from_height(HEIGHT)

        depths = propagate_depths(
            rays, float(camera[HEAD, 2]), lengths, root_joint=HEAD, up_camera=up_camera
        )
        resolved = resolve_foot_depths(rays * depths[:, None], rays, lengths)

        for joint in (LEFT_BIG_TOE, RIGHT_BIG_TOE, LEFT_HEEL, RIGHT_HEEL, LEFT_SMALL_TOE):
            assert np.linalg.norm(resolved[joint] - camera[joint]) < 0.01, joint

    def test_body_forward_matches_the_way_the_subject_is_turned(self):
        for yaw in (-60.0, 0.0, 60.0):
            play = reference_skeleton(HEIGHT)
            angle = np.radians(yaw)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])
            forward = body_forward(play @ rotation.T)
            expected = rotation @ np.array([0.0, 0.0, -1.0])
            assert np.dot(forward, expected) > 0.99


class TestEnforceBoneLengths:
    def test_repairs_a_stretched_skeleton(self):
        lengths = bone_lengths_from_height(HEIGHT)
        xyz = reference_skeleton(HEIGHT)
        stretched = xyz * 1.25

        repaired = enforce_bone_lengths(stretched, lengths, iterations=40)
        measured = measure_bone_lengths(repaired)
        errors = [abs(measured[b] - lengths[b]) for b in lengths]
        assert max(errors) < 0.01

    def test_leaves_a_correct_skeleton_alone(self):
        lengths = bone_lengths_from_height(HEIGHT)
        xyz = reference_skeleton(HEIGHT)
        assert np.allclose(enforce_bone_lengths(xyz, lengths, iterations=10), xyz, atol=1e-6)

    def test_reduces_error_monotonically(self):
        lengths = bone_lengths_from_height(HEIGHT)
        rng = np.random.default_rng(1)
        noisy = reference_skeleton(HEIGHT) + rng.normal(scale=0.05, size=(NUM_KEYPOINTS, 3))

        def total_error(points):
            measured = measure_bone_lengths(points)
            return sum(abs(measured[b] - lengths[b]) for b in lengths)

        previous = total_error(noisy)
        current = noisy
        for _ in range(5):
            current = enforce_bone_lengths(current, lengths, iterations=2)
            error = total_error(current)
            assert error <= previous + 1e-9
            previous = error

    def test_does_not_move_the_skeleton_far(self):
        """Constraint projection should correct shape, not translate the body."""
        lengths = bone_lengths_from_height(HEIGHT)
        rng = np.random.default_rng(2)
        noisy = reference_skeleton(HEIGHT) + rng.normal(scale=0.03, size=(NUM_KEYPOINTS, 3))
        repaired = enforce_bone_lengths(noisy, lengths, iterations=20)
        assert np.linalg.norm(repaired.mean(axis=0) - noisy.mean(axis=0)) < 0.02
