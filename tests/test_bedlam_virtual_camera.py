"""Tests for locally generated camera observations of real motion."""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.geom import camera_from_play, project_points
from bodytracker.skeleton import NUM_KEYPOINTS
from training.bedlam.joints import in_frame
from training.bedlam.virtual_camera import virtual_camera_sequence


def _smplx_stub(frames: int = 4) -> np.ndarray:
    """A non-degenerate 25-joint SMPL-X body with a simple stride."""
    joints = np.zeros((frames, 25, 3), dtype=np.float64)
    for index in range(25):
        joints[:, index] = [0.02 * index, 0.5 + 0.04 * index, 0.01 * index]
    joints[:, 0] = [0.0, 0.95, 0.0]
    joints[:, 7] = [-0.1, 0.05, 0.0]
    joints[:, 8] = [0.1, 0.05, 0.0]
    joints[:, 10] = [-0.1, 0.0, -0.12]
    joints[:, 11] = [0.1, 0.0, -0.12]
    joints[:, 12] = [0.0, 1.35, 0.0]
    joints[:, 15] = [0.0, 1.65, 0.0]
    joints[:, 16] = [-0.2, 1.4, 0.0]
    joints[:, 17] = [0.2, 1.4, 0.0]
    joints[:, 18] = [-0.35, 1.2, 0.0]
    joints[:, 19] = [0.35, 1.2, 0.0]
    joints[:, 20] = [-0.45, 1.0, 0.0]
    joints[:, 21] = [0.45, 1.0, 0.0]
    joints[:, 23] = [-0.04, 1.58, -0.05]
    joints[:, 24] = [0.04, 1.58, -0.05]
    joints[:, :, 0] += np.linspace(0.0, 0.1, frames)[:, None]
    return joints


def test_virtual_camera_makes_a_complete_training_sequence():
    sequence = virtual_camera_sequence(_smplx_stub(), fps=30.0, seed=4, source="unit")
    assert sequence.xyz_play.shape == (4, NUM_KEYPOINTS, 3)
    assert sequence.xy.shape == (4, NUM_KEYPOINTS, 2)
    assert sequence.device_pos.shape == (4, 3, 3)
    assert sequence.source == "unit"
    assert np.all(sequence.device_valid)
    assert np.isfinite(sequence.xy).all()


def test_projections_agree_with_the_saved_camera_calibration():
    sequence = virtual_camera_sequence(_smplx_stub(), seed=9)
    projected = camera_from_play(
        sequence.xyz_play[0], sequence.rotation_cw, sequence.translation_cw
    )
    assert np.all(projected[:, 2] > 0.1)
    assert np.allclose(sequence.xy[0], project_points(projected, sequence.intrinsics))
    visible = in_frame(sequence.xy[0], 1280, 720) & (projected[:, 2] > 0.1)
    assert np.array_equal(sequence.scores[0] > 0.5, visible)


def test_virtual_camera_rejects_bad_input():
    with pytest.raises(ValueError, match="SMPL-X"):
        virtual_camera_sequence(np.zeros((4, 10, 3)))
