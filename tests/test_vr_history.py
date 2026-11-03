"""Tests for matching camera frames to the SteamVR poses of the same moment."""

from __future__ import annotations

import time

import numpy as np

from bodytracker.geom.rotations import geodesic_angle_degrees
from bodytracker.types import DevicePose, VRState
from bodytracker.vr.history import VRHistory
from bodytracker.vr.source import RecordedVRSource


def _yaw(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _state(timestamp: float, x: float, yaw_deg: float = 0.0) -> VRState:
    return VRState(
        head=DevicePose(
            position=np.array([x, 1.6, 0.0]), rotation=_yaw(yaw_deg), timestamp=timestamp
        ),
        timestamp=timestamp,
    )


def test_interpolates_position_and_rotation_between_polls():
    history = VRHistory(None)
    history.record(_state(1.0, 0.0, 0.0))
    history.record(_state(1.1, 0.2, 40.0))

    middle = history.at(1.025)

    assert np.allclose(middle.head.position, [0.05, 1.6, 0.0])
    assert geodesic_angle_degrees(middle.head.rotation, _yaw(10.0)) < 1e-6


def test_refuses_a_pose_from_well_outside_the_recorded_span():
    """A stale pose is not the pose at this frame, so it must not be passed off as one."""
    history = VRHistory(None, tolerance_s=0.05)
    history.record(_state(1.0, 0.0))
    history.record(_state(1.1, 0.2))

    assert history.at(1.13) is not None
    assert history.at(1.3) is None
    assert history.at(0.9) is None


def test_ignores_out_of_order_polls():
    history = VRHistory(None)
    history.record(_state(1.0, 0.0))
    history.record(_state(0.5, 9.0))
    assert len(history) == 1


def test_keeps_only_the_recent_span():
    history = VRHistory(None, poll_hz=10, span_s=1.0)
    for index in range(50):
        history.record(_state(index / 10.0, float(index)))
    assert len(history) == 11
    assert history.at(0.0) is None


def test_polls_the_source_in_the_background():
    class Source:
        def poll(self):
            now = time.monotonic()
            return _state(now, 0.0)

    history = VRHistory(Source(), poll_hz=200)
    history.start()
    try:
        deadline = time.monotonic() + 2.0
        while len(history) < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        history.stop()
    assert len(history) >= 5
    assert history.at(history.latest().timestamp - 0.002) is not None


def test_recorded_source_interpolates_rotation_too():
    source = RecordedVRSource([_state(0.0, 0.0, 0.0), _state(1.0, 1.0, 90.0)])
    middle = source.at(0.5)
    assert np.allclose(middle.head.position, [0.5, 1.6, 0.0])
    assert geodesic_angle_degrees(middle.head.rotation, _yaw(45.0)) < 1e-6
