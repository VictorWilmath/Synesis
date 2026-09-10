from __future__ import annotations

import numpy as np
import pytest

from bodytracker.calib.headset_alignment import (
    align_targets_to_headset,
    fit_headset_alignment,
    headset_forward_from_rotation,
    load_headset_alignment,
    save_headset_alignment,
    source_forward_from_skeleton,
    yaw_rotation,
)
from bodytracker.skeleton import HIP, LEFT_SHOULDER, NECK, RIGHT_SHOULDER, TrackerRole
from bodytracker.types import DevicePose, TrackerTarget


def neutral_skeleton() -> np.ndarray:
    """A body facing the model convention's -z direction."""
    xyz = np.zeros((26, 3), dtype=np.float64)
    xyz[HIP] = [0.0, 0.9, 0.0]
    xyz[NECK] = [0.0, 1.4, 0.0]
    xyz[LEFT_SHOULDER] = [-0.2, 1.4, 0.0]
    xyz[RIGHT_SHOULDER] = [0.2, 1.4, 0.0]
    return xyz


def test_fits_yaw_between_model_body_and_hmd_forward():
    source = source_forward_from_skeleton(neutral_skeleton())
    hmd_rotation = yaw_rotation(np.pi / 2.0)
    headset = headset_forward_from_rotation(hmd_rotation)

    alignment = fit_headset_alignment([source] * 24, [headset] * 24, [1.674] * 24)

    assert np.allclose(alignment.rotation @ source, headset, atol=1e-9)
    assert alignment.height_m == pytest.approx(1.8)
    assert alignment.yaw_spread_deg == pytest.approx(0.0)


def test_alignment_anchors_head_relative_targets_to_the_live_hmd():
    source_head = np.array([0.0, 1.6, 0.0])
    alignment = fit_headset_alignment(
        [np.array([0.0, 0.0, -1.0])] * 3,
        [np.array([-1.0, 0.0, 0.0])] * 3,
        [1.674] * 3,
    )
    headset = DevicePose(position=np.array([2.0, 1.7, 3.0]), rotation=yaw_rotation(np.pi / 2.0))
    targets = [
        TrackerTarget(TrackerRole.HIP, np.array([0.0, 0.9, 0.0]), np.eye(3)),
        TrackerTarget(TrackerRole.LEFT_FOOT, np.array([-0.1, 0.0, 0.0]), np.eye(3)),
    ]

    aligned = align_targets_to_headset(targets, source_head, headset, alignment)

    assert np.allclose(aligned[0].position, [2.0, 1.0, 3.0])
    assert np.allclose(aligned[1].position, [2.0, 0.1, 3.1])
    assert np.allclose(aligned[0].rotation, alignment.rotation)
    # It returns new targets and never mutates the model-space source values.
    assert np.allclose(targets[0].position, [0.0, 0.9, 0.0])


def test_persists_headset_alignment(tmp_path):
    original = fit_headset_alignment(
        [np.array([0.0, 0.0, -1.0])] * 3,
        [np.array([-1.0, 0.0, 0.0])] * 3,
        [1.674] * 3,
    )
    path = tmp_path / "calibration" / "headset_alignment.json"

    save_headset_alignment(original, path)
    loaded = load_headset_alignment(path)

    assert np.allclose(loaded.rotation, original.rotation)
    assert loaded.height_m == pytest.approx(original.height_m)
    assert loaded.samples == original.samples


def test_requires_usable_neutral_pose_samples():
    with pytest.raises(ValueError, match="paired"):
        fit_headset_alignment([], [], [])
