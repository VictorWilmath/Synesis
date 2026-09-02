"""Tests for the evaluation harness.

An eval harness that quietly reports good numbers is worse than none, so most
of these deliberately break something and check the metric notices. A metric
that cannot go up is not measuring anything.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bodytracker.app.pipeline import Pipeline
from bodytracker.config import Config
from bodytracker.eval import (
    Session,
    SessionMeta,
    SessionRecorder,
    compare_report,
    compute,
    held_out_errors,
    replay,
    still_mask,
    withhold_devices,
)
from bodytracker.eval.metrics import foot_for, jitter_for
from bodytracker.geom.spaces import intrinsics_from_fov
from bodytracker.lift.anchored import AnchoredLifter
from bodytracker.lift.geometric import GeometricLifter
from bodytracker.skeleton import TrackerRole
from bodytracker.tools.scene import build_scene

INTRINSICS = intrinsics_from_fov(1280, 720, 62.0)
HEIGHT = 1.78
CAMERA_POSITION = np.array([0.5, 1.15, 2.7])
ROLES = ["hip", "left_knee", "right_knee", "left_foot", "right_foot"]


def make_scene(**kwargs):
    params = dict(
        intrinsics=INTRINSICS,
        camera_position=CAMERA_POSITION,
        height_m=HEIGHT,
        frames=150,
    )
    params.update(kwargs)
    return build_scene(**params)


def record(scene, *, height_m: float = HEIGHT, label: str = "test", calibrated: bool = True):
    """Drive a pipeline over a scene and capture the result as a Session."""
    config = Config()
    config.body.height_m = height_m
    config.osc.roles = list(ROLES)

    pipeline = Pipeline(config, intrinsics=INTRINSICS)
    if calibrated:
        pipeline.set_extrinsics(scene.rotation_cw, scene.translation_cw)

    recorder = SessionRecorder(
        SessionMeta(
            intrinsics=INTRINSICS,
            rotation_cw=scene.rotation_cw if calibrated else None,
            translation_cw=scene.translation_cw if calibrated else None,
            height_m=height_m,
            roles=list(ROLES),
            label=label,
        )
    )
    for index, frame in enumerate(scene.frames):
        timestamp = index / 30.0
        result = pipeline.process(frame.keypoints, frame.vr, timestamp=timestamp)
        recorder.add(result, latency_ms=18.0 + index % 3)
    return recorder.build()


@pytest.fixture(scope="module")
def scene():
    return make_scene()


@pytest.fixture(scope="module")
def session(scene):
    return record(scene)


class TestSessionRoundTrip:
    def test_survives_a_save_and_load(self, session, tmp_path):
        path = session.save(tmp_path / "session")
        loaded = Session.load(path)

        assert len(loaded) == len(session)
        assert loaded.meta.roles == session.meta.roles
        assert loaded.meta.height_m == pytest.approx(session.meta.height_m)
        assert np.allclose(loaded.keypoints_xy, session.keypoints_xy, equal_nan=True)
        assert np.allclose(loaded.device_positions, session.device_positions, equal_nan=True)
        assert np.allclose(loaded.target_positions, session.target_positions, equal_nan=True)
        assert np.allclose(loaded.meta.rotation_cw, session.meta.rotation_cw)

    def test_reconstructs_pipeline_inputs(self, session, scene):
        """The whole point of the format: inputs come back intact."""
        keypoints = session.keypoints_at(3)
        assert np.allclose(keypoints.xy, scene.frames[3].keypoints.xy, atol=1e-5)

        vr = session.vr_at(3)
        assert np.allclose(vr.head.position, scene.frames[3].vr.head.position, atol=1e-9)
        assert np.allclose(vr.head.rotation, scene.frames[3].vr.head.rotation, atol=1e-9)

    def test_records_dropped_frames_as_gaps(self, scene):
        config = Config()
        config.osc.roles = list(ROLES)
        pipeline = Pipeline(config, intrinsics=INTRINSICS)
        pipeline.set_extrinsics(scene.rotation_cw, scene.translation_cw)

        recorder = SessionRecorder(
            SessionMeta(intrinsics=INTRINSICS, roles=list(ROLES), label="gappy")
        )
        for index, frame in enumerate(scene.frames[:20]):
            keypoints = None if index in (5, 6) else frame.keypoints
            recorder.add(pipeline.process(keypoints, frame.vr, timestamp=index / 30.0))

        built = recorder.build()
        assert not built.has_keypoints()[5]
        assert built.has_keypoints()[4]
        assert built.tracked().sum() == 18

    def test_rejects_a_future_format(self, session, tmp_path, monkeypatch):
        path = session.save(tmp_path / "s")
        data = dict(np.load(path, allow_pickle=True))
        data["format_version"] = np.array(99)
        np.savez_compressed(path, **data)
        with pytest.raises(ValueError, match="format version"):
            Session.load(path)


class TestReplay:
    def test_reproduces_the_original_run(self, session):
        """Replaying with the same settings must give the same answer.

        If it does not, every comparison built on replay is meaningless.
        """
        again = replay(session)
        assert np.allclose(again.skeleton_xyz, session.skeleton_xyz, equal_nan=True, atol=1e-6)
        assert np.allclose(
            again.target_positions, session.target_positions, equal_nan=True, atol=1e-6
        )

    def test_withholding_hides_only_the_named_device(self, session):
        vr = session.vr_at(0)
        hidden = withhold_devices(vr, ("head",))
        assert hidden.head is None
        assert hidden.left_hand is not None
        assert vr.head is not None, "must not mutate the original"

    def test_a_different_lifter_gives_a_different_answer(self, session):
        anchored = replay(session, lifter=AnchoredLifter(), label="anchored")
        geometric = replay(session, lifter=GeometricLifter(), label="geometric")
        assert not np.allclose(
            anchored.skeleton_xyz, geometric.skeleton_xyz, equal_nan=True, atol=1e-3
        )


class TestHeldOutError:
    def test_measures_error_without_any_ground_truth(self, session):
        errors = held_out_errors(session)
        assert set(errors) == {"head", "left_hand", "right_hand"}
        for device, error in errors.items():
            assert error.frames > 100
            assert error.median_m < 0.25, f"{device}: {error}"

    def test_the_head_is_predicted_well_when_the_body_scale_is_right(self, session):
        """Pins the current camera-only head error, which is the number to beat.

        Hiding the headset drops the lifter back to guessing depth from
        apparent body size, so about thirteen centimetres is what the geometry
        alone is worth even with perfect keypoints and a correct height. This
        is the baseline the trained lifter has to improve on.
        """
        assert held_out_errors(session)["head"].median_m < 0.2

    def test_a_wrong_body_scale_shows_up(self, scene):
        """The metric has to be able to go up, or it is measuring nothing."""
        good = held_out_errors(record(scene, height_m=HEIGHT))["head"].median_m
        bad = held_out_errors(record(scene, height_m=1.45))["head"].median_m
        assert bad > good * 2.0

    def test_noisy_keypoints_show_up(self):
        clean = held_out_errors(record(make_scene()))["head"].median_m
        noisy = held_out_errors(record(make_scene(noise_px=6.0, seed=5)))["head"].median_m
        assert noisy > clean

    def test_refuses_without_calibration(self, scene):
        uncalibrated = record(scene, calibrated=False)
        with pytest.raises(ValueError, match="extrinsics"):
            held_out_errors(uncalibrated)

    def test_hiding_the_head_actually_changes_the_prediction(self, session):
        """Guards against the withholding silently not taking effect."""
        withheld = replay(session, withhold=("head",))
        assert not np.allclose(
            withheld.skeleton_xyz, session.skeleton_xyz, equal_nan=True, atol=1e-3
        )


class TestStillDetection:
    def test_finds_a_stationary_subject(self):
        still = record(make_scene(cycle_s=1e6))
        assert still_mask(still).mean() > 0.9

    def test_does_not_call_a_walking_subject_still(self):
        assert still_mask(record(make_scene(cycle_s=4.0))).mean() < 0.2


class TestJitter:
    def test_rises_with_keypoint_noise(self):
        def rms(noise):
            session = record(make_scene(cycle_s=1e6, noise_px=noise, seed=11))
            return jitter_for(session, TrackerRole.HIP, still_mask(session)).rms_speed_mm_s

        assert rms(4.0) > rms(0.5) * 2.0

    def test_a_perfectly_still_perfect_input_barely_moves(self):
        session = record(make_scene(cycle_s=1e6))
        assert jitter_for(session, TrackerRole.HIP, still_mask(session)).rms_speed_mm_s < 5.0


class TestFootMetrics:
    def test_reports_contact(self, session):
        foot = foot_for(session, TrackerRole.LEFT_FOOT)
        assert foot is not None
        assert foot.contact_fraction > 0.9

    def test_notices_a_foot_below_the_floor(self, session):
        sunk = replace(session, target_positions=session.target_positions.copy())
        index = sunk.meta.roles.index(TrackerRole.LEFT_FOOT.value)
        sunk.target_positions[:, index, 1] = -0.04

        foot = foot_for(sunk, TrackerRole.LEFT_FOOT)
        assert foot.penetration_fraction == 1.0
        assert foot.mean_penetration_mm == pytest.approx(40.0, abs=1.0)

    def test_walking_is_not_counted_as_skate(self, session):
        """A foot moving during a step is a person walking, not an error."""
        foot = foot_for(session, TrackerRole.LEFT_FOOT)
        assert foot.slide_frames == 0
        assert foot.slide_mm_per_s == 0.0

    def test_notices_a_foot_skating_while_the_user_stands_still(self):
        still = record(make_scene(cycle_s=1e6), label="still")
        drifting = replace(still, target_positions=still.target_positions.copy())
        index = drifting.meta.roles.index(TrackerRole.LEFT_FOOT.value)
        drifting.target_positions[:, index, 0] += np.linspace(0.0, 0.5, len(drifting))

        clean = foot_for(still, TrackerRole.LEFT_FOOT)
        skating = foot_for(drifting, TrackerRole.LEFT_FOOT)
        assert clean.slide_frames > 100
        assert skating.slide_mm_per_s > clean.slide_mm_per_s + 50.0

    def test_a_clean_still_session_barely_skates(self):
        still = record(make_scene(cycle_s=1e6), label="still")
        assert foot_for(still, TrackerRole.LEFT_FOOT).slide_mm_per_s < 10.0

    def test_the_pipeline_keeps_feet_out_of_the_floor(self, session):
        for role in (TrackerRole.LEFT_FOOT, TrackerRole.RIGHT_FOOT):
            assert foot_for(session, role).penetration_fraction == 0.0


class TestSessionMetrics:
    def test_reports_the_basics(self, session):
        metrics = compute(session)
        assert metrics.frames == len(session)
        assert metrics.fps == pytest.approx(30.0, rel=0.05)
        assert metrics.track_rate == 1.0
        assert metrics.latency.median_ms == pytest.approx(19.0, abs=1.5)
        assert "lift_ms" in metrics.latency.stages_median_ms

    def test_report_is_printable(self, session):
        text = compute(session).report()
        assert "tracked" in text
        assert "latency" in text

    def test_notices_dropped_tracking(self, scene):
        config = Config()
        config.osc.roles = list(ROLES)
        pipeline = Pipeline(config, intrinsics=INTRINSICS)
        pipeline.set_extrinsics(scene.rotation_cw, scene.translation_cw)
        recorder = SessionRecorder(
            SessionMeta(intrinsics=INTRINSICS, roles=list(ROLES), label="patchy")
        )
        for index, frame in enumerate(scene.frames[:40]):
            keypoints = None if index >= 20 else frame.keypoints
            recorder.add(pipeline.process(keypoints, frame.vr, timestamp=index / 30.0))

        metrics = compute(recorder.build())
        assert metrics.track_rate == pytest.approx(0.5)
        assert metrics.stale_rate > 0.0


class TestCompare:
    def test_anchoring_beats_geometry_on_held_out_wrists(self, scene):
        """The Phase 2 claim, stated as a number an eval harness can check."""
        session = record(scene, height_m=1.55)
        config = Config()
        config.body.height_m = 1.55

        anchored = held_out_errors(
            session, config, devices=("left_hand",), lifter_factory=AnchoredLifter
        )["left_hand"]
        geometric = held_out_errors(
            session, config, devices=("left_hand",), lifter_factory=GeometricLifter
        )["left_hand"]
        assert anchored.median_m < geometric.median_m

    def test_report_marks_improvements(self, session):
        better = compute(replay(session, lifter=AnchoredLifter(), label="anchored"))
        worse = compute(replay(session, lifter=GeometricLifter(), label="geometric"))
        text = compare_report(worse, better)
        assert "anchored" in text
        assert "geometric" in text
        assert "%" in text
