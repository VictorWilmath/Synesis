"""End-to-end tests for the per-frame pipeline.

Everything between "here are some 2D keypoints" and "here are tracker poses to
send to VRChat", driven from a synthetic session so no camera, headset or
onnxruntime is involved. This is the level at which the Phase 2 claim can be
checked: with a headset and a calibrated camera, the trackers land in the right
place in the room.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.app.pipeline import Pipeline
from bodytracker.config import Config
from bodytracker.geom.spaces import intrinsics_from_fov
from bodytracker.skeleton import (
    HIP,
    LEFT_ANKLE,
    LEFT_KNEE,
    RIGHT_ANKLE,
    RIGHT_KNEE,
    TrackerRole,
)
from bodytracker.tools.scene import build_scene

INTRINSICS = intrinsics_from_fov(1280, 720, 62.0)
HEIGHT = 1.78
CAMERA_POSITION = np.array([0.5, 1.15, 2.7])

ROLE_JOINT = {
    TrackerRole.HIP: HIP,
    TrackerRole.LEFT_KNEE: LEFT_KNEE,
    TrackerRole.RIGHT_KNEE: RIGHT_KNEE,
    TrackerRole.LEFT_FOOT: LEFT_ANKLE,
    TrackerRole.RIGHT_FOOT: RIGHT_ANKLE,
}


@pytest.fixture(scope="module")
def scene():
    return build_scene(
        intrinsics=INTRINSICS,
        camera_position=CAMERA_POSITION,
        height_m=HEIGHT,
        frames=150,
    )


def make_config(height_m: float = HEIGHT) -> Config:
    config = Config()
    config.body.height_m = height_m
    # The default is the three-point set; exercise the knees too, since they
    # are derived rather than measured and so are the most likely to be wrong.
    config.osc.roles = [role.value for role in ROLE_JOINT]
    return config


def make_pipeline(scene, *, height_m: float = HEIGHT, calibrated: bool = True) -> Pipeline:
    pipeline = Pipeline(make_config(height_m), intrinsics=INTRINSICS)
    if calibrated:
        pipeline.set_extrinsics(scene.rotation_cw, scene.translation_cw)
    return pipeline


def drive(pipeline: Pipeline, scene, *, with_vr: bool = True):
    results = []
    for index, frame in enumerate(scene.frames):
        results.append(
            pipeline.process(
                frame.keypoints,
                frame.vr if with_vr else None,
                timestamp=index / 30.0,
            )
        )
    return results


def tracker_errors(results, scene) -> dict[TrackerRole, float]:
    """Median distance from each tracker to the joint it represents."""
    collected: dict[TrackerRole, list[float]] = {role: [] for role in ROLE_JOINT}
    for result, frame in zip(results, scene.frames, strict=True):
        for target in result.targets:
            joint = ROLE_JOINT.get(target.role)
            if joint is None or not target.valid:
                continue
            collected[target.role].append(
                float(np.linalg.norm(target.position - frame.play_xyz[joint]))
            )
    return {role: float(np.median(values)) for role, values in collected.items() if values}


class TestTrackerAccuracy:
    def test_trackers_land_on_the_right_joints(self, scene):
        results = drive(make_pipeline(scene), scene)
        errors = tracker_errors(results, scene)

        assert set(errors) == set(ROLE_JOINT)
        for role, error in errors.items():
            # Loose next to the lifter's own four millimetres, because almost
            # all of this is the smoothing filter's deliberate lag against a
            # subject who never stops moving. See `test_lag_is_mostly_lag`.
            assert error < 0.08, f"{role} off by {error:.3f} m"

    def test_the_error_is_lag_rather_than_inaccuracy(self):
        """Distinguishes a filter that trails from a solver that is wrong.

        Worth separating, because they look identical in an averaged error
        number and have completely different fixes.
        """
        still = build_scene(
            intrinsics=INTRINSICS,
            camera_position=CAMERA_POSITION,
            height_m=HEIGHT,
            frames=150,
            cycle_s=1e6,
        )
        errors = tracker_errors(drive(make_pipeline(still), still), still)
        for role, error in errors.items():
            assert error < 0.02, f"{role} off by {error:.3f} m on a stationary subject"

    def test_the_headset_fixes_a_wrong_height_setting(self, scene):
        """Nobody types their height in correctly. It should not matter much."""
        with_vr = tracker_errors(drive(make_pipeline(scene, height_m=1.60), scene), scene)
        without_vr = tracker_errors(
            drive(make_pipeline(scene, height_m=1.60), scene, with_vr=False), scene
        )
        assert with_vr[TrackerRole.HIP] < without_vr[TrackerRole.HIP] / 2.0

    def test_every_frame_produces_a_full_set_of_trackers(self, scene):
        results = drive(make_pipeline(scene), scene)
        expected = set(make_config().osc.tracker_roles())
        for result in results:
            assert {t.role for t in result.targets} == expected

    def test_rotations_stay_orthonormal(self, scene):
        for result in drive(make_pipeline(scene), scene):
            for target in result.targets:
                rotation = target.rotation
                assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5)
                assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-5)

    def test_nothing_is_nan(self, scene):
        for result in drive(make_pipeline(scene), scene):
            for target in result.targets:
                assert np.all(np.isfinite(target.position))
                assert np.all(np.isfinite(target.rotation))


class TestPhysicalPlausibility:
    def test_feet_do_not_sink_through_the_floor(self, scene):
        config = Config()
        results = drive(make_pipeline(scene), scene)
        for result in results:
            for target in result.targets:
                if target.role in (TrackerRole.LEFT_FOOT, TrackerRole.RIGHT_FOOT):
                    assert target.position[1] > -config.solve.floor_epsilon_m - 1e-6

    def test_output_is_smoother_than_the_input(self, scene):
        """Filtering has to actually reduce jitter, not just add lag."""
        noisy = build_scene(
            intrinsics=INTRINSICS,
            camera_position=CAMERA_POSITION,
            height_m=HEIGHT,
            frames=150,
            noise_px=2.5,
            seed=7,
        )

        def hip_track(config):
            pipeline = Pipeline(config, intrinsics=INTRINSICS)
            pipeline.set_extrinsics(noisy.rotation_cw, noisy.translation_cw)
            return np.array(
                [
                    next(
                        t.position
                        for t in pipeline.process(
                            frame.keypoints, frame.vr, timestamp=index / 30.0
                        ).targets
                        if t.role is TrackerRole.HIP
                    )
                    for index, frame in enumerate(noisy.frames)
                ]
            )

        raw_config = make_config()
        raw_config.filter.min_cutoff = 1000.0
        raw_config.filter.beta = 0.0

        def jerk(track):
            return float(np.mean(np.linalg.norm(np.diff(track, n=2, axis=0), axis=1)))

        assert jerk(hip_track(make_config())) < 0.5 * jerk(hip_track(raw_config))


class TestDegradation:
    def test_holds_the_last_pose_through_a_dropout(self, scene):
        pipeline = make_pipeline(scene)
        drive(pipeline, scene)

        last = {
            t.role: t.position.copy()
            for t in pipeline.process(
                scene.frames[-1].keypoints, scene.frames[-1].vr, timestamp=2.0
            ).targets
        }

        dropped = pipeline.process(None, None, timestamp=2.05)
        assert dropped.targets
        for target in dropped.targets:
            assert target.stale
            assert np.allclose(target.position, last[target.role])

    def test_stops_reporting_after_a_long_dropout(self, scene):
        pipeline = make_pipeline(scene)
        drive(pipeline, scene)
        timeout = pipeline.config.filter.hold_last_valid_s
        assert pipeline.process(None, None, timestamp=100.0 + timeout).targets == []

    def test_runs_uncalibrated(self, scene):
        """No extrinsics yet is the normal state on first launch."""
        results = drive(make_pipeline(scene, calibrated=False), scene)
        assert all(r.targets for r in results)
        assert all(np.all(np.isfinite(t.position)) for r in results for t in r.targets)

    def test_counts_dropped_frames(self, scene):
        pipeline = make_pipeline(scene)
        drive(pipeline, scene)
        pipeline.process(None, None, timestamp=99.0)
        stats = pipeline.stats()
        assert stats["tracked"] == len(scene.frames)
        assert stats["dropped"] == 1

    def test_reset_clears_the_held_state(self, scene):
        pipeline = make_pipeline(scene)
        drive(pipeline, scene)
        pipeline.reset()
        assert pipeline.process(None, None, timestamp=2.0).targets == []


class TestDiagnostics:
    def test_reports_the_anchor_residual(self, scene):
        results = drive(make_pipeline(scene), scene)
        assert results[-1].diagnostics.get("anchored") is True
        assert results[-1].diagnostics.get("head_residual_m") is not None

    def test_reports_timings(self, scene):
        result = drive(make_pipeline(scene), scene)[-1]
        assert result.timings["lift_ms"] >= 0.0
        assert result.timings["solve_ms"] >= 0.0
