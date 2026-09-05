"""Tests for detector noise, windowing, resampling and shard I/O."""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.geom import camera_from_play, intrinsics_from_fov, play_from_camera
from bodytracker.skeleton import HEAD, HIP, LEFT_WRIST, NUM_KEYPOINTS
from bodytracker.tools.scene import build_scene
from training.bedlam.joints import OPENPOSE_BODY25, OPENPOSE_TO_HALPE
from training.bedlam.noise import KeypointNoise, default_sigma_px, fit_sigma
from training.bedlam.samples import (
    from_bedlam_frame,
    from_scene,
    load_bedlam_npz,
    load_shard,
    resample,
    save_shard,
    window_starts,
    windows,
)


def _scene(**kwargs):
    defaults = dict(
        intrinsics=intrinsics_from_fov(1280, 720, 70.0),
        frames=50,
        fps=30.0,
        cycle_s=8.0,
        seed=0,
    )
    defaults.update(kwargs)
    return build_scene(**defaults)


class TestNoise:
    def test_a_known_seed_is_deterministic(self):
        xy = np.zeros((20, NUM_KEYPOINTS, 2))
        first, _ = KeypointNoise()(xy, seed=3)
        second, _ = KeypointNoise()(xy, seed=3)
        assert np.allclose(first, second)

    def test_different_seeds_differ(self):
        xy = np.zeros((20, NUM_KEYPOINTS, 2))
        a, _ = KeypointNoise()(xy, seed=1)
        b, _ = KeypointNoise()(xy, seed=2)
        assert not np.allclose(a, b)

    def test_errors_persist_across_frames(self):
        """An AR(1) of 0.8 should be strongly autocorrelated, not white."""
        xy = np.zeros((200, NUM_KEYPOINTS, 2))
        noisy, _ = KeypointNoise(dropout=0.0, rho=0.85)(xy, seed=4)
        series = noisy[:, HIP, 0]
        corr = np.corrcoef(series[:-1], series[1:])[0, 1]
        assert corr > 0.6

    def test_a_dropped_joint_holds_still_instead_of_jumping_to_the_origin(self):
        xy = np.zeros((30, NUM_KEYPOINTS, 2))
        xy[:, LEFT_WRIST] = [400.0, 300.0]
        noise = KeypointNoise(dropout=1.0, dropout_persist=1.0, sigma_px=np.zeros(NUM_KEYPOINTS))
        noisy, scores = noise(xy, seed=0)
        assert np.all(scores[:, LEFT_WRIST] < 0.2)
        assert np.allclose(noisy[:, LEFT_WRIST], [400.0, 300.0])

    def test_fit_sigma_recovers_a_known_scale(self):
        rng = np.random.default_rng(0)
        truth = rng.normal(size=(200, NUM_KEYPOINTS, 2)) * 10
        sigma = default_sigma_px()
        predicted = truth + rng.normal(size=truth.shape) * sigma[None, :, None]
        measured = fit_sigma(predicted, truth)
        assert np.allclose(measured, sigma, rtol=0.15)


class TestResample:
    def test_a_six_fps_clip_becomes_thirty(self):
        sequence = from_scene(_scene(frames=13, fps=6.0, cycle_s=8.0))
        resampled = resample(sequence, fps=30.0)
        assert resampled.fps == pytest.approx(30.0)
        # 12/6 = 2 seconds of source → about 61 frames at 30 fps, inclusive.
        assert 55 <= len(resampled) <= 65

    def test_the_endpoints_stay_put(self):
        sequence = from_scene(_scene(frames=13, fps=6.0))
        resampled = resample(sequence, fps=30.0)
        assert np.allclose(resampled.xyz_play[0], sequence.xyz_play[0], atol=1e-6)
        assert np.allclose(resampled.xyz_play[-1], sequence.xyz_play[-1], atol=1e-6)

    def test_the_floor_stays_the_floor(self):
        sequence = from_scene(_scene(frames=13, fps=6.0))
        resampled = resample(sequence)
        feet_y = resampled.xyz_play[:, :, 1].min(axis=1)
        assert feet_y.min() > -0.05
        assert feet_y.max() < 0.15

    def test_reprojected_pixels_land_near_the_originals_at_source_times(self):
        sequence = from_scene(_scene(frames=13, fps=6.0))
        resampled = resample(sequence, fps=30.0)
        # Frame 0 is shared; the 2D should match the clean projection.
        assert np.allclose(resampled.xy[0], sequence.xy[0], atol=0.5)


class TestWindows:
    def test_a_short_clip_yields_nothing(self):
        assert window_starts(5, fps=30.0, window_s=0.9) == []

    def test_windows_have_the_advertised_length(self):
        sequence = from_scene(_scene())
        clips = windows(sequence, window_s=0.9, stride_s=0.45, noise=None)
        assert clips
        assert all(c.xy.shape == (27, NUM_KEYPOINTS, 2) for c in clips)

    def test_noise_is_applied_per_window_not_to_the_source(self):
        sequence = from_scene(_scene())
        clean = sequence.xy.copy()
        clips = windows(sequence, noise=KeypointNoise(), rng=np.random.default_rng(0))
        assert np.allclose(sequence.xy, clean)
        assert not np.allclose(clips[0].xy, sequence.xy[: clips[0].xy.shape[0]])

    def test_overlapping_windows_share_their_middle(self):
        sequence = from_scene(_scene())
        clips = windows(sequence, window_s=0.9, stride_s=0.45, noise=None)
        # 0.45 s at 30 fps is 14 frames of overlap on a 27-frame window,
        # so clip 1's start should equal clip 0's frame 14.
        stride = 14
        assert len(clips) >= 2
        assert np.allclose(clips[1].xyz_play[0], clips[0].xyz_play[stride])


class TestShard:
    def test_round_trips(self, tmp_path):
        sequence = from_scene(_scene(), source="unit")
        clips = windows(sequence, window_s=0.3, stride_s=0.15, noise=None)
        assert len(clips) >= 8
        path = save_shard(tmp_path / "shard.npz", clips[:8], notes="test")
        loaded = load_shard(path)
        assert len(loaded) == 8
        assert loaded[0].source == "unit"
        assert np.allclose(loaded[3].xyz_play, clips[3].xyz_play)
        assert np.allclose(loaded[3].device_pos, clips[3].device_pos)

    def test_refuses_an_empty_batch(self, tmp_path):
        with pytest.raises(ValueError, match="no samples"):
            save_shard(tmp_path / "empty.npz", [])


class TestFromScene:
    def test_preserves_the_head_height(self):
        scene = _scene(frames=30)
        sequence = from_scene(scene)
        # Standing head is around 1.6–1.7 m; crouch drops it, but not to the floor.
        assert sequence.xyz_play[:, HEAD, 1].min() > 1.0
        assert sequence.xyz_play[:, HEAD, 1].max() < 2.0

    def test_every_device_is_valid(self):
        sequence = from_scene(_scene(frames=20))
        assert np.all(sequence.device_valid)


class TestBedlamFrame:
    def test_a_known_skeleton_survives_the_camera_round_trip(self):
        scene = _scene(frames=1)
        frame = scene.frames[0]
        # Pretend the 2D came in as OpenPose: permute Halpe → BODY_25.
        openpose = np.full((OPENPOSE_BODY25, 2), np.nan)
        for src, dst in OPENPOSE_TO_HALPE:
            openpose[src] = frame.keypoints.xy[dst]

        # Camera-space 3D from the scene's own extrinsics.
        xyz_cam = camera_from_play(frame.play_xyz, scene.rotation_cw, scene.translation_cw)
        openpose_xyz = np.full((OPENPOSE_BODY25, 3), np.nan)
        for src, dst in OPENPOSE_TO_HALPE:
            openpose_xyz[src] = xyz_cam[dst]
        xy, xyz_play, _, vr, rotation_cw, translation_cw = from_bedlam_frame(
            openpose,
            openpose_xyz,
            scene.intrinsics,
            _cam_ext_for(scene),
            layout="openpose25",
        )
        # Play-space hips should come back near the original, within the
        # floor-origin convention (origin under the camera, not under the body).
        recovered = play_from_camera(xyz_cam, rotation_cw, translation_cw)
        # Distances between joints are what must survive, not the origin.
        orig_span = np.linalg.norm(frame.play_xyz[HEAD] - frame.play_xyz[HIP])
        new_span = np.linalg.norm(recovered[HEAD] - recovered[HIP])
        assert new_span == pytest.approx(orig_span, rel=0.02)
        assert vr.head is not None
        assert np.isfinite(xy[HIP]).all()


def _cam_ext_for(scene) -> np.ndarray:
    """A BEDLAM-style 4x4 whose rotation matches this scene's play-space camera.

    ``rotation_cw = R_bedlam @ WORLD_FROM_BEDLAM``, so
    ``R_bedlam = rotation_cw @ WORLD_FROM_BEDLAM`` since the flip is involutory.
    """
    flip = np.diag([1.0, -1.0, -1.0])
    matrix = np.eye(4)
    matrix[:3, :3] = scene.rotation_cw @ flip
    return matrix


class TestBedlamNpz:
    def test_reads_a_minimal_file(self, tmp_path):
        scene = _scene(frames=12)
        sequence = from_scene(scene)
        gtkps = np.zeros((12, 25, 3))
        xyz_cam = np.zeros((12, 25, 3))
        for i, frame in enumerate(scene.frames):
            cam = camera_from_play(frame.play_xyz, scene.rotation_cw, scene.translation_cw)
            for src, dst in OPENPOSE_TO_HALPE:
                gtkps[i, src, :2] = frame.keypoints.xy[dst]
                gtkps[i, src, 2] = 1.0
                xyz_cam[i, src] = cam[dst]
        path = tmp_path / "scene.npz"
        np.savez(
            path,
            gtkps=gtkps,
            xyz_cam=xyz_cam,
            cam_int=np.stack([scene.intrinsics] * 12),
            cam_ext=np.stack([_cam_ext_for(scene)] * 12),
        )
        loaded = load_bedlam_npz(path, layout="openpose25")
        assert len(loaded) == 12
        # Origin sits under the camera, so compare bone lengths, not points.
        truth_len = np.linalg.norm(sequence.xyz_play[0, HEAD] - sequence.xyz_play[0, HIP])
        got_len = np.linalg.norm(loaded.xyz_play[0, HEAD] - loaded.xyz_play[0, HIP])
        assert got_len == pytest.approx(truth_len, rel=0.02)
        assert np.allclose(loaded.xy[0, HIP], sequence.xy[0, HIP], atol=0.5)

    def test_refuses_a_file_with_no_3d(self, tmp_path):
        path = tmp_path / "empty.npz"
        np.savez(path, gtkps=np.zeros((4, 25, 3)), cam_int=np.eye(3), cam_ext=np.eye(4))
        with pytest.raises(KeyError, match="xyz_cam"):
            load_bedlam_npz(path)


class TestPrepCli:
    def test_raw_motion_rejects_a_negative_skip(self, tmp_path):
        from training.bedlam.prep import main

        motion = tmp_path / "motion_seq.npz"
        np.savez(motion, ignored=np.zeros(1))
        assert (
            main(
                [
                    "--motion",
                    str(motion),
                    "--smplx-models",
                    str(tmp_path),
                    "--skip",
                    "-1",
                    "--out",
                    str(tmp_path / "shards"),
                ]
            )
            == 1
        )

    def test_raw_motion_requires_model_files(self, tmp_path):
        from training.bedlam.prep import main

        motion = tmp_path / "motion_seq.npz"
        np.savez(motion, ignored=np.zeros(1))
        assert main(["--motion", str(motion), "--out", str(tmp_path / "shards")]) == 1

    def test_synthetic_writes_a_shard(self, tmp_path):
        from training.bedlam.prep import main

        out = tmp_path / "shards"
        assert main([
            "--synthetic",
            "--out",
            str(out),
            "--frames",
            "30",
            "--window",
            "0.3",
            "--stride",
            "0.15",
            "--shard-size",
            "4",
            "--no-noise",
        ]) == 0
        shards = list(out.glob("shard_*.npz"))
        assert shards
        loaded = load_shard(shards[0])
        assert loaded[0].xy.shape[1] == NUM_KEYPOINTS
