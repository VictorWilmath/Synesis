from __future__ import annotations

import numpy as np

from bodytracker.app.runtime import (
    _accept_extrinsics,
    _extrapolate_targets,
    _osc_alignment_head,
    _rebase_uncalibrated_targets,
)
from bodytracker.config import load
from bodytracker.skeleton import HEAD, NUM_KEYPOINTS, TrackerRole
from bodytracker.types import DevicePose, FrameResult, Skeleton3D, TrackerTarget


def test_uncalibrated_osc_alignment_uses_lifted_head_not_steamvr_head():
    skeleton = Skeleton3D(
        xyz=np.arange(NUM_KEYPOINTS * 3, dtype=np.float32).reshape(NUM_KEYPOINTS, 3),
        scores=np.ones(NUM_KEYPOINTS, dtype=np.float32),
    )
    result = FrameResult(timestamp=12.0, skeleton3d=skeleton)
    steamvr_head = DevicePose(position=np.array([9.0, 9.0, 9.0]), rotation=np.eye(3))

    head = _osc_alignment_head(result, steamvr_head, calibrated=False)

    assert head is not None
    assert np.array_equal(head.position, skeleton.xyz[HEAD])
    assert np.array_equal(head.rotation, np.eye(3))


def test_calibrated_osc_alignment_keeps_real_headset_pose():
    result = FrameResult(timestamp=12.0)
    steamvr_head = DevicePose(position=np.array([1.0, 2.0, 3.0]), rotation=np.eye(3))

    assert _osc_alignment_head(result, steamvr_head, calibrated=True) is steamvr_head


def test_rebases_rough_targets_around_the_live_headset():
    xyz = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    xyz[HEAD] = [0.4, 1.6, -2.0]
    result = FrameResult(
        timestamp=12.0,
        skeleton3d=Skeleton3D(xyz=xyz, scores=np.ones(NUM_KEYPOINTS, dtype=np.float32)),
        targets=[
            TrackerTarget(
                role=TrackerRole.HIP,
                position=np.array([0.4, 0.9, -2.0]),
                rotation=np.eye(3),
            )
        ],
    )
    headset = DevicePose(position=np.array([2.0, 1.7, 4.0]), rotation=np.eye(3))

    rebased = _rebase_uncalibrated_targets(result, headset)

    assert np.allclose(rebased[0].position, [2.0, 1.0, 4.0])
    assert np.allclose(result.targets[0].position, [0.4, 0.9, -2.0])


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


def test_default_runtime_uses_automatic_headset_alignment():
    config = load(local=False)
    assert config.osc.send_head is True
    assert config.osc.rebase_uncalibrated_to_head is True


def test_default_runtime_uses_the_low_impact_vrchat_path():
    config = load(local=False)
    assert (config.camera.width, config.camera.height, config.camera.fps) == (640, 480, 30)
    assert config.camera.mirror is True
    assert config.pose2d.mode == "lightweight"
    assert config.pose2d.device == "cpu"
    assert config.pose2d.detect_interval == 60
    assert config.calibration.online_refine is False
    assert config.lift.method == "neural"
    assert config.lift.device == "cpu"


def test_rejects_high_error_room_calibration():
    class Extrinsics:
        rms_error_px = 19.2

    assert _accept_extrinsics(Extrinsics(), 12.0) is None
