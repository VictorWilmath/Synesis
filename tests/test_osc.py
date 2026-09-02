"""Tests for the VRChat OSC wire format.

These lock down the contract described in VRChat's OSC trackers docs: metres in
Unity space, Euler ZXY in degrees, slots numbered 1-8.
"""

from __future__ import annotations

import numpy as np
import pytest

from bodytracker.geom import matrix_from_euler_zxy_degrees, rotation_from_forward_up
from bodytracker.osc import (
    HEAD_POSITION_ADDRESS,
    HEAD_ROTATION_ADDRESS,
    VRChatOSCSender,
    position_address,
    rotation_address,
)
from bodytracker.skeleton import TrackerRole, assign_slots
from bodytracker.tools.synthetic import PATTERNS, SyntheticBody
from bodytracker.types import DevicePose, TrackerTarget


class FakeClient:
    def __init__(self) -> None:
        self.messages: list[tuple[str, list[float]]] = []

    def send_message(self, address: str, value: list[float]) -> None:
        self.messages.append((address, list(value)))

    def by_address(self, address: str) -> list[float]:
        for addr, value in self.messages:
            if addr == address:
                return value
        raise KeyError(address)

    @property
    def addresses(self) -> list[str]:
        return [a for a, _ in self.messages]


def make_target(role: TrackerRole, position, rotation=None) -> TrackerTarget:
    return TrackerTarget(
        role=role,
        position=np.asarray(position, dtype=np.float64),
        rotation=np.eye(3) if rotation is None else rotation,
    )


# ---------------------------------------------------------------------------
# Addresses and slots
# ---------------------------------------------------------------------------


class TestAddresses:
    def test_format(self):
        assert position_address(1) == "/tracking/trackers/1/position"
        assert rotation_address(8) == "/tracking/trackers/8/rotation"

    @pytest.mark.parametrize("slot", [0, -1, 9, 100])
    def test_rejects_out_of_range(self, slot):
        """There is no tracker 0, and VRChat supports at most 8."""
        with pytest.raises(ValueError):
            position_address(slot)
        with pytest.raises(ValueError):
            rotation_address(slot)


class TestSlotAssignment:
    def test_is_one_indexed_and_ordered(self):
        slots = assign_slots([TrackerRole.HIP, TrackerRole.LEFT_FOOT, TrackerRole.RIGHT_FOOT])
        assert slots == {
            TrackerRole.HIP: 1,
            TrackerRole.LEFT_FOOT: 2,
            TrackerRole.RIGHT_FOOT: 3,
        }

    def test_rejects_duplicates(self):
        with pytest.raises(ValueError, match="duplicate"):
            assign_slots([TrackerRole.HIP, TrackerRole.HIP])

    def test_rejects_more_than_eight(self):
        with pytest.raises(ValueError, match="at most 8"):
            assign_slots(list(TrackerRole) + [TrackerRole.HIP])

    def test_accepts_exactly_eight(self):
        assert len(assign_slots(list(TrackerRole))) == 8


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


class TestWireFormat:
    def test_position_flips_z_for_unity(self):
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=0)
        sender.send([make_target(TrackerRole.HIP, [0.1, 0.9, -2.0])])
        assert client.by_address(position_address(1)) == pytest.approx([0.1, 0.9, 2.0])

    def test_sends_three_floats_per_message(self):
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=0)
        sender.send([make_target(TrackerRole.HIP, [1.0, 2.0, 3.0])])
        for _, value in client.messages:
            assert len(value) == 3
            assert all(isinstance(v, float) for v in value)

    def test_facing_steamvr_forward_is_unity_identity(self):
        """A subject facing SteamVR-forward must emit zero rotation."""
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=0)
        rotation = rotation_from_forward_up(np.array([0.0, 0.0, -1.0]))
        sender.send([make_target(TrackerRole.HIP, [0, 1, 0], rotation)])
        assert client.by_address(rotation_address(1)) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)

    def test_turning_left_in_play_space_yields_positive_unity_yaw(self):
        """Handedness flip inverts yaw sense; this pins the direction down."""
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=0)
        # Facing play -x, i.e. the subject's left when facing SteamVR-forward.
        rotation = rotation_from_forward_up(np.array([-1.0, 0.0, 0.0]))
        sender.send([make_target(TrackerRole.HIP, [0, 1, 0], rotation)])
        _, yaw, _ = client.by_address(rotation_address(1))
        assert yaw == pytest.approx(-90.0, abs=1e-6)

    def test_rotation_is_euler_zxy_degrees(self):
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=0)
        # Build in Unity space, invert the flip to get the play-space input.
        expected = [12.0, -34.0, 56.0]
        unity = matrix_from_euler_zxy_degrees(*expected)
        flip = np.diag([1.0, 1.0, -1.0])
        sender.send([make_target(TrackerRole.HIP, [0, 0, 0], flip @ unity @ flip)])
        assert client.by_address(rotation_address(1)) == pytest.approx(expected, abs=1e-6)

    def test_emits_position_and_rotation_per_tracker(self):
        client = FakeClient()
        roles = [TrackerRole.HIP, TrackerRole.LEFT_FOOT, TrackerRole.RIGHT_FOOT]
        sender = VRChatOSCSender(roles, client=client, rate_hz=0)
        sender.send([make_target(r, [0, 1, 0]) for r in roles])
        assert len(client.messages) == 6
        assert set(client.addresses) == {
            position_address(1),
            rotation_address(1),
            position_address(2),
            rotation_address(2),
            position_address(3),
            rotation_address(3),
        }


class TestHeadEndpoint:
    def test_omitted_by_default(self):
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=0)
        head = DevicePose(position=np.array([0.0, 1.6, 0.0]), rotation=np.eye(3))
        sender.send([make_target(TrackerRole.HIP, [0, 1, 0])], head=head)
        assert HEAD_POSITION_ADDRESS not in client.addresses

    def test_sent_when_enabled(self):
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=0, send_head=True)
        head = DevicePose(position=np.array([0.0, 1.6, -0.2]), rotation=np.eye(3))
        sender.send([make_target(TrackerRole.HIP, [0, 1, 0])], head=head)
        assert client.by_address(HEAD_POSITION_ADDRESS) == pytest.approx([0.0, 1.6, 0.2])
        assert HEAD_ROTATION_ADDRESS in client.addresses

    def test_invalid_head_is_skipped(self):
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=0, send_head=True)
        head = DevicePose(position=np.zeros(3), rotation=np.eye(3), valid=False)
        sender.send([make_target(TrackerRole.HIP, [0, 1, 0])], head=head)
        assert HEAD_POSITION_ADDRESS not in client.addresses


class TestSendBehaviour:
    def test_unassigned_roles_are_skipped(self):
        """Estimating a joint but not sending it must not raise."""
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=0)
        sender.send(
            [
                make_target(TrackerRole.HIP, [0, 1, 0]),
                make_target(TrackerRole.LEFT_ELBOW, [0, 1, 0]),
            ]
        )
        assert len(client.messages) == 2

    def test_rate_limit_drops_early_frames(self):
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=60)
        targets = [make_target(TrackerRole.HIP, [0, 1, 0])]
        assert sender.send(targets, now=0.0) is True
        assert sender.send(targets, now=0.001) is False
        assert sender.send(targets, now=0.020) is True
        assert sender.frames_sent == 2

    def test_force_bypasses_rate_limit(self):
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=60)
        targets = [make_target(TrackerRole.HIP, [0, 1, 0])]
        sender.send(targets, now=0.0)
        assert sender.send(targets, now=0.001, force=True) is True

    def test_stale_targets_are_still_sent(self):
        """A slightly stale tracker beats one that teleports to the origin."""
        client = FakeClient()
        sender = VRChatOSCSender([TrackerRole.HIP], client=client, rate_hz=0)
        target = make_target(TrackerRole.HIP, [0, 1, 0])
        target.stale = True
        sender.send([target])
        assert len(client.messages) == 2


# ---------------------------------------------------------------------------
# Synthetic motion
# ---------------------------------------------------------------------------


class TestSyntheticBody:
    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_produces_all_eight_roles(self, pattern):
        targets = SyntheticBody().pose(0.37, pattern)
        assert {t.role for t in targets} == set(TrackerRole)

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_values_are_finite_and_plausible(self, pattern):
        body = SyntheticBody(height_m=1.75)
        for t in np.linspace(0, 4, 60):
            for target in body.pose(float(t), pattern):
                assert np.all(np.isfinite(target.position))
                assert np.all(np.isfinite(target.rotation))
                assert -0.05 <= target.position[1] <= 1.75, target.role
                assert np.linalg.norm(target.position[[0, 2]]) < 1.5

    @pytest.mark.parametrize("pattern", PATTERNS)
    def test_rotations_are_valid(self, pattern):
        body = SyntheticBody()
        for t in np.linspace(0, 4, 40):
            for target in body.pose(float(t), pattern):
                rot = target.rotation
                assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
                assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-9)

    def test_anatomical_ordering_holds(self):
        """Feet below knees below hips below chest, at every instant."""
        body = SyntheticBody()
        for pattern in PATTERNS:
            for t in np.linspace(0, 4, 40):
                pose = {tgt.role: tgt.position for tgt in body.pose(float(t), pattern)}
                assert pose[TrackerRole.LEFT_FOOT][1] < pose[TrackerRole.LEFT_KNEE][1]
                assert pose[TrackerRole.LEFT_KNEE][1] < pose[TrackerRole.HIP][1]
                assert pose[TrackerRole.HIP][1] < pose[TrackerRole.CHEST][1]

    def test_static_pattern_does_not_move(self):
        body = SyntheticBody()
        first = body.pose(0.0, "static")
        later = body.pose(3.3, "static")
        for a, b in zip(first, later, strict=True):
            assert np.allclose(a.position, b.position)

    def test_bob_actually_moves(self):
        body = SyntheticBody()
        heights = [
            next(t for t in body.pose(float(x), "bob") if t.role == TrackerRole.HIP).position[1]
            for x in np.linspace(0, 2, 20)
        ]
        assert max(heights) - min(heights) > 0.1

    def test_spin_sweeps_full_yaw(self):
        """The spin pattern must cover the whole yaw range, wrapping included."""
        from bodytracker.geom import euler_zxy_degrees_from_matrix, unity_rotation_from_play

        body = SyntheticBody(period_s=2.0)
        yaws = []
        for t in np.linspace(0, 2, 48, endpoint=False):
            target = next(x for x in body.pose(float(t), "spin") if x.role == TrackerRole.HIP)
            yaws.append(euler_zxy_degrees_from_matrix(unity_rotation_from_play(target.rotation))[1])
        assert max(yaws) > 150 and min(yaws) < -150

    def test_spin_rotates_positions_with_facing(self):
        """Positions and orientation must stay consistent, or the avatar shears."""
        body = SyntheticBody(period_s=4.0)
        quarter = body.pose(1.0, "spin")
        elbows = {t.role: t.position for t in quarter}
        left = elbows[TrackerRole.LEFT_ELBOW]
        # A quarter turn moves the left elbow from -x onto the z axis.
        assert abs(left[0]) < 1e-9
        assert abs(abs(left[2]) - 0.150) < 1e-9
