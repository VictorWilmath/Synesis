from __future__ import annotations

import numpy as np

from bodytracker.app.runtime import (
    _OSCOutputInterpolator,
    describe_placement,
)
from bodytracker.config import load
from bodytracker.geom import geodesic_angle_degrees
from bodytracker.skeleton import TrackerRole
from bodytracker.types import DevicePose, TrackerTarget


class _Sender:
    def __init__(self):
        self.sent = []

    def send(self, targets, head=None, *, now=None, force=False):
        self.sent.append(targets)


def _hip(x: float, rotation: np.ndarray | None = None) -> list[TrackerTarget]:
    return [
        TrackerTarget(
            TrackerRole.HIP,
            np.array([x, 1.0, 0.0]),
            np.eye(3) if rotation is None else rotation,
        )
    ]


def _hip_x(output: _OSCOutputInterpolator, now: float) -> float:
    targets, _ = output._snapshot(now)
    return float(targets[0].position[0])


def test_eases_from_one_webcam_pose_to_the_next():
    output = _OSCOutputInterpolator(_Sender(), rate_hz=60)
    output.publish(_hip(0.0), None, 1.0, arrived_at=1.05)
    output.publish(_hip(0.1), None, 1.1, arrived_at=1.15)

    assert np.isclose(_hip_x(output, 1.15), 0.0)
    assert np.isclose(_hip_x(output, 1.20), 0.05)
    assert np.isclose(_hip_x(output, 1.25), 0.1)
    # Never past the newest pose, however long the next one takes.
    assert np.isclose(_hip_x(output, 3.0), 0.1)


def test_an_early_pose_carries_on_from_where_the_output_is():
    output = _OSCOutputInterpolator(_Sender(), rate_hz=60)
    output.publish(_hip(0.0), None, 1.0, arrived_at=1.05)
    output.publish(_hip(0.1), None, 1.1, arrived_at=1.15)
    output.publish(_hip(0.2), None, 1.2, arrived_at=1.20)

    assert np.isclose(_hip_x(output, 1.20), 0.05)
    assert np.isclose(_hip_x(output, 1.25), 0.125)
    assert np.isclose(_hip_x(output, 1.30), 0.2)


def test_jumps_straight_to_a_pose_after_a_gap():
    output = _OSCOutputInterpolator(_Sender(), rate_hz=60)
    output.publish(_hip(0.0), None, 1.0, arrived_at=1.05)
    output.publish(_hip(0.5), None, 3.0, arrived_at=3.05)

    assert np.isclose(_hip_x(output, 3.05), 0.5)


def test_eases_rotation_too():
    quarter_turn = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    output = _OSCOutputInterpolator(_Sender(), rate_hz=60)
    output.publish(_hip(0.0), None, 1.0, arrived_at=1.05)
    output.publish(_hip(0.0, quarter_turn), None, 1.1, arrived_at=1.15)

    targets, _ = output._snapshot(1.20)
    assert np.isclose(geodesic_angle_degrees(targets[0].rotation, np.eye(3)), 45.0)


def test_cleared_output_sends_nothing_until_the_next_pose():
    output = _OSCOutputInterpolator(_Sender(), rate_hz=60)
    output.publish(_hip(0.0), None, 1.0)
    assert output._snapshot(1.0) is not None
    output.clear()
    assert output._snapshot(1.0) is None


def test_placement_is_described_relative_to_the_headset():
    # SteamVR faces -z, so identity is looking forward along -z, with +x to the right.
    head = DevicePose(position=np.array([0.0, 1.6, 0.0]), rotation=np.eye(3))
    targets = [
        TrackerTarget(TrackerRole.HIP, np.array([0.0, 0.9, -0.1]), np.eye(3)),
        TrackerTarget(TrackerRole.LEFT_FOOT, np.array([-0.12, 0.05, 0.02]), np.eye(3)),
    ]
    lines = describe_placement(targets, head)
    assert lines[0] == "hip 0.90m up, +0.10 ahead, +0.00 right"
    assert lines[1] == "left_foot 0.05m up, -0.02 ahead, -0.12 right"
    assert describe_placement(targets, None) == []


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
