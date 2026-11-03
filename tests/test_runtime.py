from __future__ import annotations

import numpy as np

from bodytracker.app.runtime import _extrapolate_targets, _OSCOutputInterpolator
from bodytracker.config import load
from bodytracker.skeleton import TrackerRole
from bodytracker.types import TrackerTarget


def test_extrapolates_slow_webcam_tracker_positions_for_osc_output():
    before = [
        TrackerTarget(TrackerRole.HIP, np.array([0.0, 1.0, 0.0]), np.eye(3)),
    ]
    latest = [
        TrackerTarget(TrackerRole.HIP, np.array([0.1, 1.0, 0.0]), np.eye(3)),
    ]

    predicted = _extrapolate_targets(latest, 1.0, before, 0.9, 1.05)

    assert np.allclose(predicted[0].position, [0.15, 1.0, 0.0])
    assert np.allclose(latest[0].position, [0.1, 1.0, 0.0])


def test_stops_prediction_after_short_safety_horizon():
    target = TrackerTarget(TrackerRole.HIP, np.array([0.1, 1.0, 0.0]), np.eye(3))
    prior = TrackerTarget(TrackerRole.HIP, np.array([0.0, 1.0, 0.0]), np.eye(3))

    predicted = _extrapolate_targets([target], 1.0, [prior], 0.9, 2.0)

    # The 100 ms horizon prevents an old pose from flying away when tracking pauses.
    assert np.allclose(predicted[0].position, [0.2, 1.0, 0.0])


def test_cleared_output_sends_nothing_until_the_next_pose():
    class Sender:
        def __init__(self):
            self.sent = []

        def send(self, targets, head=None, *, now=None, force=False):
            self.sent.append(targets)

    output = _OSCOutputInterpolator(Sender(), rate_hz=60)
    output.publish(
        [TrackerTarget(TrackerRole.HIP, np.array([0.0, 1.0, 0.0]), np.eye(3))], None, 1.0
    )
    assert output._snapshot(1.0) is not None
    output.clear()
    assert output._snapshot(1.0) is None


def test_default_runtime_calibrates_itself():
    config = load(local=False)
    assert config.calibration.path == "calibration/camera.json"
    assert config.calibration.min_samples > 0
    assert config.osc.send_head is False


def test_default_runtime_uses_the_low_impact_vrchat_path():
    config = load(local=False)
    assert (config.camera.width, config.camera.height, config.camera.fps) == (640, 480, 30)
    assert config.camera.mirror is True
    assert config.pose2d.mode == "lightweight"
    assert config.pose2d.device == "cpu"
    assert config.pose2d.detect_interval == 60
    assert config.lift.method == "neural"
    assert config.lift.device == "cpu"
