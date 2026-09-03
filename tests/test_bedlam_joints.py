"""Tests for BEDLAM joint-layout conversion onto Halpe26.

The permutations are the whole point: a silent swap of left and right would
train a lifter that puts the wrong foot on the floor, and every accuracy
number measured against it would be a lie. Tests are written against the
documented OpenPose and SMPL-X index lists, not against a round-trip through
the same mapping.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.skeleton import (
    HEAD,
    HIP,
    LEFT_ANKLE,
    LEFT_EAR,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NECK,
    NOSE,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_EAR,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from bodytracker.tools.synthetic import reference_skeleton
from training.bedlam.joints import (
    OPENPOSE_BODY25,
    OPENPOSE_TO_HALPE,
    detect_layout,
    from_openpose_body25,
    from_smplx_body,
    in_frame,
    to_halpe,
)


def _halpe_to_openpose(xyz: np.ndarray) -> np.ndarray:
    out = np.full((OPENPOSE_BODY25, xyz.shape[-1]), np.nan, dtype=np.float64)
    for src, dst in OPENPOSE_TO_HALPE:
        out[src] = xyz[dst]
    return out


class TestOpenPose:
    def test_round_trips_every_shared_joint(self):
        truth = reference_skeleton(1.75)
        recovered = from_openpose_body25(_halpe_to_openpose(truth))
        for _, dst in OPENPOSE_TO_HALPE:
            assert np.allclose(recovered[dst], truth[dst])

    def test_fills_head_from_the_ears(self):
        truth = reference_skeleton(1.75)
        recovered = from_openpose_body25(_halpe_to_openpose(truth))
        expected = 0.5 * (truth[LEFT_EAR] + truth[RIGHT_EAR])
        assert np.allclose(recovered[HEAD], expected)

    def test_does_not_swap_left_and_right(self):
        """The single most expensive permutation error this file can make."""
        truth = reference_skeleton(1.75)
        recovered = from_openpose_body25(_halpe_to_openpose(truth))
        assert recovered[LEFT_WRIST, 0] < recovered[RIGHT_WRIST, 0]
        assert recovered[LEFT_ANKLE, 0] < recovered[RIGHT_ANKLE, 0]
        assert recovered[LEFT_SHOULDER, 0] < recovered[RIGHT_SHOULDER, 0]

    def test_accepts_a_batch_and_trailing_columns(self):
        truth = np.stack([reference_skeleton(1.7), reference_skeleton(1.8)])
        openpose = np.stack([_halpe_to_openpose(t) for t in truth])
        padded = np.concatenate([openpose, np.zeros((2, 20, 3))], axis=1)
        recovered = from_openpose_body25(padded)
        assert recovered.shape == (2, NUM_KEYPOINTS, 3)
        assert np.allclose(recovered[0, NECK], truth[0, NECK])

    def test_rejects_a_short_layout(self):
        with pytest.raises(ValueError, match="at least 25"):
            from_openpose_body25(np.zeros((10, 3)))


class TestSmpx:
    def test_copies_the_joints_that_exist(self):
        smplx = np.zeros((25, 3))
        smplx[12] = [0.0, 1.4, 0.0]  # neck
        smplx[15] = [0.0, 1.6, 0.0]  # head
        smplx[20] = [-0.3, 1.0, 0.0]  # left wrist
        smplx[21] = [0.3, 1.0, 0.0]  # right wrist
        smplx[23] = [-0.03, 1.55, -0.08]  # left eye
        smplx[24] = [0.03, 1.55, -0.08]  # right eye
        recovered = from_smplx_body(smplx)
        assert np.allclose(recovered[NECK], smplx[12])
        assert np.allclose(recovered[HEAD], smplx[15])
        assert np.allclose(recovered[LEFT_WRIST], smplx[20])
        assert recovered[LEFT_WRIST, 0] < recovered[RIGHT_WRIST, 0]

    def test_puts_the_nose_between_the_eyes(self):
        smplx = np.zeros((25, 3))
        smplx[23] = [-0.04, 1.55, -0.1]
        smplx[24] = [0.04, 1.55, -0.1]
        recovered = from_smplx_body(smplx)
        assert np.allclose(recovered[NOSE], [0.0, 1.55, -0.1])

    def test_ears_sit_on_the_correct_sides_of_the_head(self):
        smplx = np.zeros((25, 3))
        smplx[15] = [0.0, 1.6, 0.0]
        smplx[16] = [-0.2, 1.4, 0.0]  # left shoulder
        smplx[17] = [0.2, 1.4, 0.0]
        recovered = from_smplx_body(smplx)
        assert recovered[LEFT_EAR, 0] < recovered[HEAD, 0] < recovered[RIGHT_EAR, 0]


class TestDetect:
    def test_a_long_row_is_bedlam2_openpose(self):
        assert detect_layout(np.zeros((80, 3))) == "openpose25"

    def test_a_twenty_five_joint_row_is_openpose(self):
        assert detect_layout(np.zeros((25, 3))) == "openpose25"

    def test_to_halpe_dispatches(self):
        truth = reference_skeleton(1.75)
        openpose = _halpe_to_openpose(truth)
        assert np.allclose(to_halpe(openpose)[HIP], truth[HIP])

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError, match="unrecognised"):
            detect_layout(np.zeros((3, 3)))


class TestInFrame:
    def test_a_centred_point_counts(self):
        assert in_frame(np.array([[640.0, 360.0]]), 1280, 720)[0]

    def test_a_point_off_the_edge_does_not(self):
        assert not in_frame(np.array([[-1.0, 360.0]]), 1280, 720)[0]
        assert not in_frame(np.array([[640.0, 800.0]]), 1280, 720)[0]

    def test_nan_is_out(self):
        assert not in_frame(np.array([[np.nan, 360.0]]), 1280, 720)[0]
