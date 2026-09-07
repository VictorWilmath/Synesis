from __future__ import annotations

import numpy as np

from bodytracker.app.runtime import _osc_alignment_head
from bodytracker.skeleton import HEAD, NUM_KEYPOINTS
from bodytracker.types import DevicePose, FrameResult, Skeleton3D


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
