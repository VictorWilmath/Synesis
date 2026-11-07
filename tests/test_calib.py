"""Tests for automatic camera calibration from the skeleton and SteamVR.

The headline test is simple: place a camera with an unknown lens in a synthetic
room, let its frames arrive late the way a webcam's do, and check the
calibrator finds the camera, the lens and the delay with no setup step.
Everything else measures how that holds up against noise, outliers, a mirrored
image and a camera that gets knocked.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bodytracker.calib.anchors import ANCHORS
from bodytracker.calib.auto import (
    AutoCalibrator,
    CameraCalibration,
    SampleCollector,
    SampleSet,
    assess,
    focal_from_fov,
    load_calibration,
    save_calibration,
    solve_camera,
)
from bodytracker.config import CalibrationConfig
from bodytracker.geom.rotations import geodesic_angle_degrees
from bodytracker.geom.spaces import intrinsics_from_fov
from bodytracker.skeleton import HEAD, LEFT_WRIST, RIGHT_WRIST
from bodytracker.tools.scene import build_scene, vr_state_for
from bodytracker.types import Keypoints2D
from bodytracker.vr.history import VRHistory

WIDTH, HEIGHT = 640, 480
TRUE_FOV = 74.0
INTRINSICS = intrinsics_from_fov(WIDTH, HEIGHT, TRUE_FOV)
CAMERA_POSITION = np.array([0.6, 1.15, 2.6])
FPS = 30.0
LATENCY = 0.08
CONFIG = CalibrationConfig()
LAGS = np.arange(0.0, CONFIG.max_latency_s + 1e-9, CONFIG.latency_step_s)


def make_scene(seconds: float = 12.0, **kwargs):
    kwargs.setdefault("camera_position", CAMERA_POSITION)
    return build_scene(intrinsics=INTRINSICS, frames=int(seconds * FPS), fps=FPS, **kwargs)


def history_for(scene) -> VRHistory:
    history = VRHistory(None, poll_hz=FPS, span_s=1e6)
    for frame in scene.frames:
        history.record(frame.vr)
    return history


def camera_frames(scene, *, latency: float = LATENCY, mirror: bool = False):
    """Keypoints as a webcam delivers them: stamped when they arrive, not when exposed."""
    for frame in scene.frames:
        xy = frame.keypoints.xy.copy()
        scores = frame.keypoints.scores.copy()
        if mirror:
            xy[:, 0] = (WIDTH - 1) - xy[:, 0]
            xy[[LEFT_WRIST, RIGHT_WRIST]] = xy[[RIGHT_WRIST, LEFT_WRIST]]
            scores[[LEFT_WRIST, RIGHT_WRIST]] = scores[[RIGHT_WRIST, LEFT_WRIST]]
        yield Keypoints2D(xy=xy, scores=scores, timestamp=frame.keypoints.timestamp + latency)


def collect(scene, *, min_spacing_m: float = 0.04, **kwargs) -> SampleCollector:
    history = history_for(scene)
    collector = SampleCollector(LAGS, min_spacing_m=min_spacing_m)
    for keypoints in camera_frames(scene, **kwargs):
        collector.add(keypoints, history.at_many([keypoints.timestamp - lag for lag in LAGS]))
    return collector


def solve(collector: SampleCollector, **kwargs) -> CameraCalibration | None:
    return solve_camera(collector.snapshot(), WIDTH, HEIGHT, **kwargs)


def truth(scene) -> CameraCalibration:
    return CameraCalibration(
        rotation_cw=scene.rotation_cw,
        translation_cw=scene.translation_cw,
        focal_px=focal_from_fov(WIDTH, TRUE_FOV),
        width=WIDTH,
        height=HEIGHT,
        latency_s=LATENCY,
        offsets={a.name: a.local_offset.copy() for a in ANCHORS},
        rms_px=1.0,
        inliers=300,
        total=300,
        position_std_m=0.01,
        focal_std=0.01,
    )


@pytest.fixture(scope="module")
def clean_scene():
    return make_scene()


@pytest.fixture(scope="module")
def noisy_scene():
    return make_scene(seconds=16.0, noise_px=3.0, seed=1)


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

    def test_in_place_motion_fits_a_small_play_space(self):
        scene = make_scene(seconds=8.0, motion="in_place")
        heads = np.array([f.play_xyz[HEAD] for f in scene.frames])
        spread = heads.max(axis=0) - heads.min(axis=0)
        assert spread[0] < 0.5 and spread[2] < 0.5

    def test_everything_is_in_front_of_the_camera(self, clean_scene):
        from bodytracker.geom import camera_from_play

        for frame in clean_scene.frames:
            camera = camera_from_play(
                frame.play_xyz, clean_scene.rotation_cw, clean_scene.translation_cw
            )
            assert np.all(camera[:, 2] > 0)

    def test_offsets_override_is_honoured(self):
        from bodytracker.tools.synthetic import reference_skeleton

        xyz = reference_skeleton(1.75)
        custom = {a.name: a.local_offset + 0.05 for a in ANCHORS}
        state = vr_state_for(xyz, 0.0, 0.0, custom)
        recovered = state.head.position + state.head.rotation @ custom["head"]
        assert np.allclose(recovered, xyz[HEAD], atol=1e-9)


class TestSampleCollector:
    def test_collects_three_anchors_per_kept_frame(self, clean_scene):
        collector = collect(clean_scene)
        counts = collector.counts()
        assert counts["head"] == counts["left_hand"] == counts["right_hand"] == collector.frames
        assert len(collector) == 3 * collector.frames

    def test_rejects_repeats_of_a_stationary_subject(self, clean_scene):
        """Repeating one correspondence adds no information, so it is dropped."""
        frame = clean_scene.frames[0]
        states = [frame.vr] * len(LAGS)
        collector = SampleCollector(LAGS, min_spacing_m=0.05)
        assert collector.add(frame.keypoints, states) == 3
        for _ in range(50):
            collector.add(frame.keypoints, states)
        assert len(collector) == 3
        assert collector.rejected_close == 50

    def test_skips_unconfident_keypoints(self, clean_scene):
        frame = clean_scene.frames[0]
        keypoints = Keypoints2D(xy=frame.keypoints.xy.copy(), scores=frame.keypoints.scores.copy())
        keypoints.scores[HEAD] = 0.01
        collector = SampleCollector(LAGS, min_score=0.5)
        assert collector.add(keypoints, [frame.vr] * len(LAGS)) == 2
        assert collector.counts()["head"] == 0

    def test_needs_the_headset_at_every_candidate_latency(self, clean_scene):
        frame = clean_scene.frames[0]
        states = [frame.vr] * (len(LAGS) - 1) + [None]
        assert SampleCollector(LAGS).add(frame.keypoints, states) == 0

    def test_records_the_opposite_wrist_for_the_mirror_check(self, clean_scene):
        frame = clean_scene.frames[0]
        collector = SampleCollector(LAGS)
        collector.add(frame.keypoints, [frame.vr] * len(LAGS))
        samples = collector.snapshot()
        left = int(np.flatnonzero(samples.anchor == 1)[0])
        assert np.allclose(samples.pixel[left], frame.keypoints.xy[LEFT_WRIST])
        assert np.allclose(samples.mirror_pixel[left], frame.keypoints.xy[RIGHT_WRIST])

    def test_drops_the_oldest_frames_when_full(self, clean_scene):
        history = history_for(clean_scene)
        collector = SampleCollector(LAGS, min_spacing_m=0.001, max_samples=30)
        for keypoints in camera_frames(clean_scene):
            collector.add(keypoints, history.at_many([keypoints.timestamp - lag for lag in LAGS]))
        assert len(collector) <= 30
        assert collector.frames == len(collector) // 3

    def test_clear_resets_everything(self, clean_scene):
        collector = collect(clean_scene)
        collector.clear()
        assert len(collector) == 0 and collector.frames == 0


class TestSolveCamera:
    def test_recovers_camera_lens_and_delay_from_clean_data(self, clean_scene):
        """Starting from a 62 degree guess, with frames arriving 80 ms late."""
        result = solve(collect(clean_scene), initial_fov_deg=62.0)
        assert result is not None
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.01
        assert geodesic_angle_degrees(result.rotation_cw, clean_scene.rotation_cw) < 0.3
        assert result.fov_deg == pytest.approx(TRUE_FOV, abs=0.3)
        assert result.latency_s == pytest.approx(LATENCY, abs=0.005)
        assert result.rms_px < 0.5
        assert assess(result, CONFIG) is None

    def test_survives_webcam_keypoint_noise(self, noisy_scene):
        result = solve(collect(noisy_scene))
        assert result is not None
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.10
        assert geodesic_angle_degrees(result.rotation_cw, noisy_scene.rotation_cw) < 1.5
        assert result.fov_deg == pytest.approx(TRUE_FOV, abs=2.0)
        assert result.latency_s == pytest.approx(LATENCY, abs=0.02)

    def test_finds_a_long_webcam_delay(self):
        """A webcam in a dim room on auto exposure delivered frames over 200 ms late."""
        scene = make_scene(seconds=16.0, noise_px=3.0, motion="in_place", seed=7)
        result = solve(collect(scene, latency=0.3))
        assert result is not None
        assert result.latency_s == pytest.approx(0.3, abs=0.02)
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.10
        assert assess(result, CONFIG) is None

    def test_reports_an_honest_uncertainty(self, noisy_scene):
        """The acceptance gate trusts this number, so it must track the real error."""
        result = solve(collect(noisy_scene))
        error = float(np.linalg.norm(result.camera_position - CAMERA_POSITION))
        assert error < 3.0 * result.position_std_m
        assert result.position_std_m < 0.10

    def test_more_movement_means_less_uncertainty(self):
        short = solve(collect(make_scene(seconds=6.0, noise_px=3.0, seed=2)))
        long = solve(collect(make_scene(seconds=16.0, noise_px=3.0, seed=2)))
        assert long.position_std_m < short.position_std_m

    def test_rejects_gross_outliers(self, clean_scene):
        """Frames where the pose model put a wrist on the wrong thing."""
        collector = collect(clean_scene)
        rng = np.random.default_rng(0)
        for index in rng.choice(len(collector), size=len(collector) // 10, replace=False):
            collector._pixel[index] = collector._pixel[index] + rng.normal(scale=150.0, size=2)
        result = solve(collector)
        assert result is not None
        assert result.inliers < result.total
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.03

    def test_recovers_device_offsets_that_differ_from_nominal(self):
        """A different headset, or different hair, moves the head keypoint."""
        true_offsets = {a.name: a.local_offset.copy() for a in ANCHORS}
        true_offsets["head"] = true_offsets["head"] + np.array([0.0, 0.03, 0.06])
        result = solve(collect(make_scene(offsets=true_offsets)))
        assert np.linalg.norm(result.offsets["head"] - true_offsets["head"]) < 0.01
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.02

    @pytest.mark.parametrize(
        "position",
        [
            np.array([0.0, 1.2, 3.0]),  # straight ahead
            np.array([2.2, 1.0, 2.2]),  # corner
            np.array([0.0, 2.2, 2.0]),  # high, looking down
        ],
    )
    def test_recovers_the_camera_wherever_it_is(self, position):
        scene = make_scene(seconds=8.0, camera_position=position)
        result = solve(collect(scene))
        assert result is not None
        assert np.linalg.norm(result.camera_position - position) < 0.03

    def test_refuses_when_there_is_too_little_data(self):
        assert solve(collect(make_scene(seconds=0.2))) is None

    def test_a_mirrored_hypothesis_does_not_fit_a_normal_camera(self, clean_scene):
        samples = collect(clean_scene).snapshot()
        mirrored = solve_camera(samples.mirrored(WIDTH), WIDTH, HEIGHT)
        assert mirrored is None or assess(mirrored, CONFIG) is not None

    def test_a_mirrored_image_fits_once_unmirrored(self, clean_scene):
        samples = collect(clean_scene, mirror=True).snapshot()
        normal = solve_camera(samples, WIDTH, HEIGHT)
        unmirrored = solve_camera(samples.mirrored(WIDTH), WIDTH, HEIGHT)
        assert normal is None or assess(normal, CONFIG) is not None
        assert assess(unmirrored, CONFIG) is None
        assert np.linalg.norm(unmirrored.camera_position - CAMERA_POSITION) < 0.02


def drive(calibrator: AutoCalibrator, scene, *, until: str | None = None, mirror: bool = False):
    """Feed a scene frame by frame, waiting out any background solve as it goes.

    `mirror` is whether the webcam's own stream is mirrored. The calibrator's
    mirror setting flips it back, as the runtime's webcam does.
    """
    updates = []
    streams = zip(camera_frames(scene), camera_frames(scene, mirror=True), strict=True)
    for plain, mirrored in streams:
        keypoints = mirrored if mirror != calibrator.mirror else plain
        calibrator.observe(keypoints)
        update = calibrator.update()
        calibrator.close()
        if update is not None:
            updates.append(update)
            if update.kind == until:
                break
    return updates


class TestAutoCalibrator:
    def test_calibrates_itself_from_normal_movement(self):
        scene = make_scene(seconds=16.0, noise_px=1.5, seed=3)
        calibrator = AutoCalibrator(CONFIG, WIDTH, HEIGHT, history_for(scene), background=False)
        updates = drive(calibrator, scene, until="calibrated")

        assert [u.kind for u in updates] == ["calibrated"]
        result = calibrator.current
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.05
        assert result.latency_s == pytest.approx(LATENCY, abs=0.02)

    def test_calibrates_without_walking_around(self):
        """A small play space: standing on one spot, reaching, turning and crouching."""
        camera = np.array([0.3, 1.3, 2.0])
        scene = make_scene(
            seconds=30.0, noise_px=3.0, motion="in_place", camera_position=camera, seed=5
        )
        calibrator = AutoCalibrator(CONFIG, WIDTH, HEIGHT, history_for(scene), background=False)
        updates = drive(calibrator, scene, until="calibrated")

        assert [u.kind for u in updates] == ["calibrated"]
        assert np.linalg.norm(calibrator.current.camera_position - camera) < 0.10
        assert calibrator.current.fov_deg == pytest.approx(TRUE_FOV, abs=2.0)

    def test_says_what_it_needs_while_calibrating(self, clean_scene):
        calibrator = AutoCalibrator(
            CONFIG, WIDTH, HEIGHT, history_for(clean_scene), background=False
        )
        assert "reach around" in calibrator.status_lines()[0]

    def test_never_calibrates_from_someone_standing_still(self, clean_scene):
        frame = clean_scene.frames[0]
        history = VRHistory(None, poll_hz=FPS, span_s=1e6)
        for index in range(300):
            history.record(replace(frame.vr, timestamp=index / FPS))
        calibrator = AutoCalibrator(CONFIG, WIDTH, HEIGHT, history, background=False)
        for index in range(300):
            calibrator.observe(replace(frame.keypoints, timestamp=index / FPS + LATENCY))
            assert calibrator.update() is None
        assert calibrator.current is None
        assert len(calibrator.collector) == 3

    def test_flips_a_mirrored_camera(self):
        scene = make_scene(seconds=16.0, noise_px=1.0, seed=4)
        config = replace(CONFIG, min_samples=90)
        calibrator = AutoCalibrator(config, WIDTH, HEIGHT, history_for(scene), background=False)
        updates = drive(calibrator, scene, mirror=True, until="mirrored")
        assert [u.kind for u in updates] == ["mirrored"]
        assert calibrator.mirror is True
        assert len(calibrator.collector) >= config.min_samples

    def test_flips_a_mirrored_camera_without_walking_around(self, tmp_path):
        scene = make_scene(seconds=40.0, noise_px=3.0, motion="in_place", seed=6)
        dump = tmp_path / "samples.npz"
        calibrator = AutoCalibrator(
            CONFIG, WIDTH, HEIGHT, history_for(scene), background=False, samples_path=dump
        )
        updates = drive(calibrator, scene, mirror=True, until="calibrated")
        assert [u.kind for u in updates] == ["mirrored", "calibrated"]
        assert dump.exists()
        result = calibrator.current
        assert result.mirror is True
        assert np.linalg.norm(result.camera_position - CAMERA_POSITION) < 0.15

    def test_keeps_a_calibration_that_still_fits(self, clean_scene):
        calibrator = AutoCalibrator(
            CONFIG,
            WIDTH,
            HEIGHT,
            history_for(clean_scene),
            current=truth(clean_scene),
            background=False,
        )
        kinds = [u.kind for u in drive(calibrator, clean_scene)]
        assert "lost" not in kinds
        assert calibrator.current is not None

    def test_notices_a_moved_camera(self, clean_scene):
        moved = make_scene(camera_position=np.array([-1.2, 1.4, 2.4]))
        calibrator = AutoCalibrator(
            CONFIG,
            WIDTH,
            HEIGHT,
            history_for(moved),
            current=truth(clean_scene),
            background=False,
        )
        kinds = [u.kind for u in drive(calibrator, moved)]
        assert kinds[0] == "lost"

    def test_runs_the_solve_in_the_background(self):
        scene = make_scene(seconds=16.0, noise_px=1.5, seed=3)
        calibrator = AutoCalibrator(CONFIG, WIDTH, HEIGHT, history_for(scene))
        updates = drive(calibrator, scene, until="calibrated")
        assert updates[-1].kind == "calibrated"


class TestPersistence:
    def test_samples_roundtrip(self, clean_scene, tmp_path):
        samples = collect(clean_scene).snapshot()
        path = tmp_path / "samples.npz"
        samples.save(path, width=WIDTH, height=HEIGHT, mirror=True)
        loaded, width, height, mirror = SampleSet.load(path)
        assert (width, height, mirror) == (WIDTH, HEIGHT, True)
        assert np.array_equal(loaded.anchor, samples.anchor)
        assert np.allclose(loaded.position, samples.position)
        assert np.allclose(loaded.mirror_pixel, samples.mirror_pixel, equal_nan=True)

    def test_roundtrip(self, clean_scene, tmp_path):
        original = solve(collect(clean_scene))
        original.mirror = True
        path = tmp_path / "camera.json"
        save_calibration(original, path)
        loaded = load_calibration(path, WIDTH, HEIGHT)

        assert np.allclose(loaded.rotation_cw, original.rotation_cw)
        assert np.allclose(loaded.translation_cw, original.translation_cw)
        assert loaded.focal_px == pytest.approx(original.focal_px)
        assert loaded.latency_s == pytest.approx(original.latency_s)
        assert loaded.mirror is True
        for name in original.offsets:
            assert np.allclose(loaded.offsets[name], original.offsets[name])

    def test_scales_to_another_resolution_with_the_same_aspect(self, clean_scene, tmp_path):
        path = tmp_path / "camera.json"
        save_calibration(truth(clean_scene), path)
        loaded = load_calibration(path, 1280, 960)
        assert loaded.focal_px == pytest.approx(2 * focal_from_fov(WIDTH, TRUE_FOV))
        assert loaded.fov_deg == pytest.approx(TRUE_FOV)

    def test_starts_over_when_the_aspect_ratio_changes(self, clean_scene, tmp_path):
        path = tmp_path / "camera.json"
        save_calibration(truth(clean_scene), path)
        assert load_calibration(path, 1280, 720) is None

    def test_missing_or_unreadable_file_returns_none(self, tmp_path):
        assert load_calibration(tmp_path / "nope.json", WIDTH, HEIGHT) is None
        broken = tmp_path / "broken.json"
        broken.write_text('{"width": 640}')
        assert load_calibration(broken, WIDTH, HEIGHT) is None

    def test_creates_the_directory(self, clean_scene, tmp_path):
        path = tmp_path / "nested" / "dir" / "camera.json"
        save_calibration(truth(clean_scene), path)
        assert path.exists()
