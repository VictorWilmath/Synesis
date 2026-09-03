"""Tests for synthesising headset and controller poses from a skeleton.

The load-bearing invariant is the same one the runtime calibrator uses: a
device pose plus its local offset must land on the observed keypoint. If that
round-trip is broken, a model trained on these poses will be asked at runtime
about a different geometry than it ever saw.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.calib.anchors import ANCHOR_BY_NAME
from bodytracker.skeleton import HEAD, LEFT_WRIST, RIGHT_WRIST
from bodytracker.tools.scene import posed_skeleton
from bodytracker.tools.synthetic import reference_skeleton
from training.bedlam.devices import from_skeleton, head_rotation, pack, unpack


class TestOffsetRoundTrip:
    def test_the_headset_offset_lands_on_the_head(self):
        xyz = reference_skeleton(1.75)
        vr = from_skeleton(xyz)
        head = vr.head
        assert head is not None
        recovered = head.position + head.rotation @ ANCHOR_BY_NAME["head"].local_offset
        assert np.allclose(recovered, xyz[HEAD])

    def test_the_controllers_land_on_the_wrists(self):
        xyz = posed_skeleton(1.75, position=np.zeros(3), yaw_deg=30.0, crouch=0.15)
        vr = from_skeleton(xyz)
        for name, joint in (("left_hand", LEFT_WRIST), ("right_hand", RIGHT_WRIST)):
            pose = getattr(vr, name)
            recovered = pose.position + pose.rotation @ ANCHOR_BY_NAME[name].local_offset
            assert np.allclose(recovered, xyz[joint])

    def test_custom_offsets_are_honoured(self):
        xyz = reference_skeleton(1.80)
        offsets = {
            "head": np.array([0.0, 0.02, 0.08]),
            "left_hand": np.array([0.01, 0.0, 0.04]),
            "right_hand": np.array([-0.01, 0.0, 0.04]),
        }
        vr = from_skeleton(xyz, offsets=offsets)
        recovered = vr.head.position + vr.head.rotation @ offsets["head"]
        assert np.allclose(recovered, xyz[HEAD])


class TestHeadOrientation:
    def test_a_subject_facing_forward_has_a_headset_facing_forward(self):
        """Play-space forward is -Z; the headset's local -Z should point that way."""
        xyz = reference_skeleton(1.75)
        rotation = head_rotation(xyz)
        local_forward = rotation @ np.array([0.0, 0.0, -1.0])
        assert local_forward[2] == pytest.approx(-1.0, abs=0.15)
        assert abs(local_forward[0]) < 0.15

    def test_turning_the_body_turns_the_headset(self):
        left = posed_skeleton(1.75, position=np.zeros(3), yaw_deg=-40.0)
        right = posed_skeleton(1.75, position=np.zeros(3), yaw_deg=40.0)
        left_fwd = head_rotation(left) @ np.array([0.0, 0.0, -1.0])
        right_fwd = head_rotation(right) @ np.array([0.0, 0.0, -1.0])
        assert left_fwd[0] * right_fwd[0] < 0
        assert abs(left_fwd[0] - right_fwd[0]) > 0.8

    def test_head_rotation_is_actually_a_rotation(self):
        rotation = head_rotation(reference_skeleton(1.75))
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)


class TestHands:
    def test_the_two_controllers_are_not_the_same_pose(self):
        xyz = reference_skeleton(1.75)
        vr = from_skeleton(xyz)
        assert not np.allclose(vr.left_hand.position, vr.right_hand.position)
        assert vr.left_hand.position[0] < vr.right_hand.position[0]


class TestPack:
    def test_round_trips(self):
        vr = from_skeleton(reference_skeleton(1.75), timestamp=1.5)
        restored = unpack(*pack(vr), timestamp=1.5)
        assert np.allclose(restored.head.position, vr.head.position)
        assert np.allclose(restored.left_hand.rotation, vr.left_hand.rotation)
        assert restored.timestamp == pytest.approx(1.5)

    def test_a_missing_device_stays_missing(self):
        vr = from_skeleton(reference_skeleton(1.75))
        vr.left_hand = None
        pos, rot, valid = pack(vr)
        assert not valid[1]
        assert unpack(pos, rot, valid).left_hand is None
