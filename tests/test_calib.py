"""Tests for HMD-anchored camera calibration.

The headline test is simple: place a camera in a synthetic room, generate what
SteamVR and the pose model would report, and check the calibrator finds the
camera we put there. Everything else measures how that degrades with noise,
dropouts and wrong assumptions.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.calib.anchors import (
    ANCHORS,
    CorrespondenceBuffer,
    build_point_arrays,
    pack_offsets,
    unpack_offsets,
)
from bodytracker.calib.extrinsics import (
    load_extrinsics,
    save_extrinsics,
    solve_extrinsics,
)
from bodytracker.geom.rotations import geodesic_angle_degrees
from bodytracker.geom.spaces import intrinsics_from_fov
from bodytracker.skeleton import HEAD
from bodytracker.tools.scene import build_scene, vr_state_for

INTRINSICS = intrinsics_from_fov(1280, 720, 62.0)
CAMERA_POSITION = np.array([0.6, 1.15, 2.6])


@pytest.fixture(scope="module")
def clean_scene():
    return build_scene(intrinsics=INTRINSICS, camera_position=CAMERA_POSITION, frames=80)


def collect(scene, buffer: CorrespondenceBuffer | None = None) -> CorrespondenceBuffer:
    buffer = buffer if buffer is not None else CorrespondenceBuffer(min_spacing_m=0.02)
    for frame in scene.frames:
        buffer.add(frame.vr, frame.keypoints)
    return buffer


class TestSceneFixture:
    def test_vr_state_reproduces_the_anchor_keypoints(self, clean_scene):
        """The scene must invert the anchor model exactly, or nothing else holds."""
        frame = clean_scene.frames[0]
        head = frame.vr.head
        recovered = head.position + head.rotation @ ANCHORS[0].local_offset
        assert np.allclose(recovered, frame.play_xyz[HEAD], atol=1e-9)

    def test_subject_moves_around(self, clean_scene):
        heads = np.array([f.play_xyz[HEAD] for f in clean_scene.frames])
        assert np.linalg.norm(heads.max(axis=0) - heads.min(axis=0)) > 1.0

    def test_everything_is_in_front_of_the_camera(self, clean_scene):
        from bodytracker.geom import camera_from_play

        for frame in clean_scene.frames:
            camera = camera_from_play(
                frame.play_xyz, clean_scene.rotation_cw, clean_scene.translation_cw
            )
            assert np.all(camera[:, 2] > 0)


class TestCorrespondenceBuffer:
    def test_collects_three_anchors_per_accepted_frame(self):
        scene = build_scene(intrinsics=INTRINSICS, frames=20)
        buffer = collect(scene)
        counts = buffer.counts()
        assert counts["head"] == counts["left_hand"] == counts["right_hand"]
        assert len(buffer) == 3 * counts["head"]

    def test_rejects_samples_from_a_stationary_subject(self):
        """Repeating one correspondence adds no information, so it is dropped."""
        scene = build_scene(intrinsics=INTRINSICS, frames=20)
        buffer = CorrespondenceBuffer(min_spacing_m=0.05)
        frame = scene.frames[0]
        first = buffer.add(frame.vr, frame.keypoints)
        for _ in range(50):
            buffer.add(frame.vr, frame.keypoints)
        assert first == 3
        assert len(buffer) == 3
        assert buffer.rejected_close == 50

    def test_skips_unconfident_keypoints(self):
        scene = build_scene(intrinsics=INTRINSICS, frames=10)
        frame = scene.frames[0]
        frame.keypoints.scores[HEAD] = 0.01
        buffer = CorrespondenceBuffer(min_spacing_m=0.01, min_score=0.5)
        assert buffer.add(frame.vr, frame.keypoints) == 2
        assert buffer.counts()["head"] == 0

    def test_ignores_frames_without_a_headset(self):
        scene = build_scene(intrinsics=INTRINSICS, frames=10)
        frame = scene.frames[0]
        frame.vr.head = None
        assert CorrespondenceBuffer().add(frame.vr, frame.keypoints) == 0

    def test_coverage_reflects_how_far_the_subject_moved(self, clean_scene):
        assert collect(clean_scene).coverage() > 1.0

    def test_coverage_is_zero_for_a_single_sample(self):
        scene = build_scene(intrinsics=INTRINSICS, frames=5)
        buffer = CorrespondenceBuffer()
        buffer.add(scene.frames[0].vr, scene.frames[0].keypoints)
        assert buffer.coverage() == 0.0

    def test_respects_the_sample_cap(self):
        scene = build_scene(intrinsics=INTRINSICS, frames=200)
        buffer = collect(scene, CorrespondenceBuffer(min_spacing_m=0.001, max_samples=30))
        assert len(buffer) <= 30 + len(ANCHORS)

    def test_clear_resets_everything(self, clean_scene):
        buffer = collect(clean_scene)
        buffer.clear()
        assert len(buffer) == 0 and buffer.coverage() == 0.0


class TestOffsetPacking:
    def test_roundtrip(self):
        offsets = {a.name: a.local_offset for a in ANCHORS}
        assert all(
            np.allclose(unpack_offsets(pack_offsets(offsets))[name], offsets[name])
            for name in offsets
        )

    def test_offsets_are_applied_in_the_device_frame(self):
        """A rotated device must carry its offset around with it."""
        from bodytracker.calib.anchors import Correspondence

        offset = np.array([0.0, 0.0, 0.10])
        upright = Correspondence("head", np.zeros(3), np.eye(3), np.zeros(2), 1.0, 0.0)
        assert np.allclose(upright.world_point(offset), [0.0, 0.0, 0.10])

        turned = Correspondence(
            "head",
            np.zeros(3),
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
            np.zeros(2),
            1.0,
            0.0,
        )
        assert np.allclose(turned.world_point(offset), [0.10, 0.0, 0.0])

    def test_build_point_arrays_shapes(self, clean_scene):
        buffer = collect(clean_scene)
        object_points, image_points = build_point_arrays(buffer.samples)
        assert object_points.shape == (len(buffer), 3)
        assert image_points.shape == (len(buffer), 2)


class TestSolveExtrinsics:
    def test_recovers_the_camera_from_perfect_data(self, clean_scene):
        result = solve_extrinsics(collect(clean_scene).samples, INTRINSICS)
        assert result is not None
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.01
        assert geodesic_angle_degrees(result.rotation_cw, clean_scene.rotation_cw) < 0.5
        assert result.rms_error_px < 1.0

    def test_solves_from_headset_only(self, clean_scene):
        """A user may not have tracked controllers in view during setup."""
        samples = [sample for sample in collect(clean_scene).samples if sample.anchor == "head"]
        result = solve_extrinsics(samples, INTRINSICS, refine_offsets=False)
        assert result is not None
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.01
        assert result.rms_error_px < 1.0

    def test_survives_realistic_keypoint_noise(self):
        """Three pixels of jitter is about what a 720p webcam gives you."""
        scene = build_scene(
            intrinsics=INTRINSICS, camera_position=CAMERA_POSITION, frames=120, noise_px=3.0
        )
        result = solve_extrinsics(collect(scene).samples, INTRINSICS)
        assert result is not None
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.15
        assert geodesic_angle_degrees(result.rotation_cw, scene.rotation_cw) < 3.0

    def test_survives_dropped_keypoints(self):
        scene = build_scene(
            intrinsics=INTRINSICS,
            camera_position=CAMERA_POSITION,
            frames=150,
            noise_px=2.0,
            dropout=0.25,
        )
        result = solve_extrinsics(collect(scene).samples, INTRINSICS)
        assert result is not None
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.2

    def test_rejects_gross_outliers(self):
        """A few frames where the keypoint landed on the wrong thing."""
        scene = build_scene(
            intrinsics=INTRINSICS, camera_position=CAMERA_POSITION, frames=120, noise_px=1.0
        )
        buffer = collect(scene)
        rng = np.random.default_rng(0)
        for index in rng.choice(len(buffer), size=len(buffer) // 10, replace=False):
            buffer.samples[index].pixel += rng.normal(scale=150.0, size=2)

        result = solve_extrinsics(buffer.samples, INTRINSICS)
        assert result is not None
        assert result.inliers < result.total
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.15

    def test_refines_a_wrong_device_offset(self):
        """Nominal offsets are guesses; the solve should recover the real ones.

        The scene is generated with the headset 8 cm further back and 3 cm
        higher than the calibrator assumes, which is the kind of error a
        different headset or a different hairstyle produces.
        """
        true_offsets = {a.name: a.local_offset.copy() for a in ANCHORS}
        true_offsets["head"] = true_offsets["head"] + np.array([0.0, 0.03, 0.08])

        scene = build_scene(
            intrinsics=INTRINSICS,
            camera_position=CAMERA_POSITION,
            frames=120,
            offsets=true_offsets,
        )
        samples = collect(scene).samples

        without = solve_extrinsics(samples, INTRINSICS, refine_offsets=False)
        with_refinement = solve_extrinsics(samples, INTRINSICS, refine_offsets=True)

        assert with_refinement.rms_error_px < without.rms_error_px
        assert np.linalg.norm(with_refinement.offsets["head"] - true_offsets["head"]) < 0.01
        assert np.linalg.norm(with_refinement.camera_position - CAMERA_POSITION) < 0.005
        assert np.linalg.norm(with_refinement.camera_position - CAMERA_POSITION) < np.linalg.norm(
            without.camera_position - CAMERA_POSITION
        )

    def test_a_wrong_offset_does_not_get_its_anchor_discarded(self):
        """The reason the refinement sees every sample, not just the inliers.

        A wrong head offset makes every head correspondence reproject badly, so
        RANSAC drops all of them, and a refinement that only looks at survivors
        has nothing left that could tell it the head offset is wrong. The
        headset is the anchor most likely to be miscalibrated and the one whose
        depth the whole tracker hangs off, so losing it is the worst case.
        """
        true_offsets = {a.name: a.local_offset.copy() for a in ANCHORS}
        true_offsets["head"] = true_offsets["head"] + np.array([0.0, 0.03, 0.08])

        scene = build_scene(
            intrinsics=INTRINSICS,
            camera_position=CAMERA_POSITION,
            frames=120,
            offsets=true_offsets,
        )
        samples = collect(scene).samples

        without = solve_extrinsics(samples, INTRINSICS, refine_offsets=False)
        with_refinement = solve_extrinsics(samples, INTRINSICS, refine_offsets=True)

        head_samples = sum(1 for s in samples if s.anchor == "head")
        assert head_samples > 0
        # Without refinement the head anchor is rejected wholesale.
        assert without.inliers <= len(samples) - head_samples
        assert with_refinement.inliers == len(samples)

    def test_offset_refinement_needs_head_rotation(self):
        """Without head articulation, offset and camera height are inseparable.

        A headset that only ever yaws keeps its local +y aligned with world +y,
        so raising the anchor and lowering the camera produce identical images.
        The optimiser then drives the residual to zero on the wrong answer.
        Documented rather than fixed, because the fix is to ask the user to
        look around, which the calibration UI already does.
        """
        true_offsets = {a.name: a.local_offset.copy() for a in ANCHORS}
        true_offsets["head"] = true_offsets["head"] + np.array([0.0, 0.03, 0.08])

        still = build_scene(
            intrinsics=INTRINSICS,
            camera_position=CAMERA_POSITION,
            frames=120,
            offsets=true_offsets,
            head_motion=False,
        )
        result = solve_extrinsics(collect(still).samples, INTRINSICS)

        # The fit is essentially perfect and the answer is still wrong.
        assert result.rms_error_px < 0.01
        assert np.linalg.norm(result.offsets["head"] - true_offsets["head"]) > 0.005

    def test_refuses_when_there_is_too_little_data(self):
        scene = build_scene(intrinsics=INTRINSICS, frames=2)
        assert solve_extrinsics(collect(scene).samples, INTRINSICS) is None

    @pytest.mark.parametrize(
        "position",
        [
            np.array([0.0, 1.2, 3.0]),  # straight ahead
            np.array([2.2, 1.0, 2.2]),  # corner
            np.array([0.0, 2.4, 2.0]),  # high, looking down
            np.array([-1.8, 0.7, 1.5]),  # low and off to one side
        ],
    )
    def test_recovers_the_camera_wherever_it_is(self, position):
        scene = build_scene(intrinsics=INTRINSICS, camera_position=position, frames=90)
        result = solve_extrinsics(collect(scene).samples, INTRINSICS)
        assert result is not None
        assert np.linalg.norm(result.camera_position - position) < 0.02

    def test_camera_position_and_forward_are_consistent(self, clean_scene):
        result = solve_extrinsics(collect(clean_scene).samples, INTRINSICS)
        to_subject = np.array([0.0, 0.95, 0.0]) - result.camera_position
        to_subject /= np.linalg.norm(to_subject)
        assert float(result.forward @ to_subject) > 0.99


class TestPersistence:
    def test_roundtrip(self, clean_scene, tmp_path):
        original = solve_extrinsics(collect(clean_scene).samples, INTRINSICS)
        path = tmp_path / "extrinsics.json"
        save_extrinsics(original, path)
        loaded = load_extrinsics(path)

        assert np.allclose(loaded.rotation_cw, original.rotation_cw)
        assert np.allclose(loaded.translation_cw, original.translation_cw)
        assert loaded.rms_error_px == pytest.approx(original.rms_error_px)
        for name in original.offsets:
            assert np.allclose(loaded.offsets[name], original.offsets[name])

    def test_missing_file_returns_none(self, tmp_path):
        assert load_extrinsics(tmp_path / "nope.json") is None

    def test_creates_the_directory(self, clean_scene, tmp_path):
        result = solve_extrinsics(collect(clean_scene).samples, INTRINSICS)
        path = tmp_path / "nested" / "dir" / "extrinsics.json"
        save_extrinsics(result, path)
        assert path.exists()


class TestVrStateHelper:
    def test_offsets_override_is_honoured(self):
        from bodytracker.tools.synthetic import reference_skeleton

        xyz = reference_skeleton(1.75)
        custom = {a.name: a.local_offset + 0.05 for a in ANCHORS}
        state = vr_state_for(xyz, 0.0, 0.0, custom)
        recovered = state.head.position + state.head.rotation @ custom["head"]
        assert np.allclose(recovered, xyz[HEAD], atol=1e-9)
