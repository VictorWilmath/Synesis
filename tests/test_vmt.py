"""Tests for the Virtual Motion Tracker output.

These lock down VMT's ``/VMT/Raw/Driver`` contract: index and mode as ints,
a time offset, then position and an x, y, z, w quaternion in SteamVR's raw
driver space, which differs from standing space by the room setup. And the
room matrix, without which VMT marks every pose invalid.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bodytracker.app.runtime import make_sender
from bodytracker.config import OSCConfig
from bodytracker.geom import matrix_from_quaternion, rotation_from_forward_up
from bodytracker.osc import (
    RAW_DRIVER_ADDRESS,
    VMTSender,
    VRChatOSCSender,
    device_name,
    find_vmt_driver,
)
from bodytracker.osc.vmt import DISABLED, ROOM_MATRIX_ADDRESS, TRACKER, VIVE_TRACKER
from bodytracker.skeleton import TrackerRole
from bodytracker.types import TrackerTarget

ROLES = [TrackerRole.HIP, TrackerRole.LEFT_FOOT, TrackerRole.RIGHT_FOOT]


class FakeClient:
    def __init__(self) -> None:
        self.messages: list[tuple[str, list]] = []

    def send_message(self, address: str, value: list) -> None:
        self.messages.append((address, list(value)))

    @property
    def poses(self) -> list[list]:
        return [value for address, value in self.messages if address == RAW_DRIVER_ADDRESS]

    @property
    def rooms(self) -> list[list]:
        return [value for address, value in self.messages if address == ROOM_MATRIX_ADDRESS]


def make_target(role: TrackerRole, position, rotation=None) -> TrackerTarget:
    return TrackerTarget(
        role=role,
        position=np.asarray(position, dtype=np.float64),
        rotation=np.eye(3) if rotation is None else rotation,
    )


def yaw(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def unpack(message: list) -> tuple[int, int, float, np.ndarray, np.ndarray]:
    index, mode, offset = message[:3]
    position = np.array(message[3:6])
    qx, qy, qz, qw = message[6:10]
    return index, mode, offset, position, matrix_from_quaternion(np.array([qw, qx, qy, qz]))


class TestWireFormat:
    def test_sends_one_raw_driver_message_per_tracker(self):
        client = FakeClient()
        sender = VMTSender(ROLES, client=client)
        sender.send([make_target(role, [0.0, 1.0, 0.0]) for role in ROLES], force=True)
        assert len(client.poses) == 3

    def test_index_and_mode_are_ints_and_the_rest_floats(self):
        client = FakeClient()
        VMTSender(ROLES, client=client).send([make_target(TrackerRole.HIP, [0.1, 0.9, -0.2])])
        message = client.poses[0]
        assert len(message) == 10
        assert all(type(v) is int for v in message[:2])
        assert all(type(v) is float for v in message[2:])
        assert message[2] == 0.0

    def test_encodes_as_vmt_expects(self):
        from pythonosc.osc_message import OscMessage
        from pythonosc.osc_message_builder import OscMessageBuilder

        client = FakeClient()
        VMTSender(ROLES, client=client).send([make_target(TrackerRole.HIP, [0.1, 0.9, -0.2])])
        builder = OscMessageBuilder(address=RAW_DRIVER_ADDRESS)
        for value in client.poses[0]:
            builder.add_arg(value)
        decoded = OscMessage(builder.build().dgram)
        assert b",iifffffff" in builder.build().dgram
        assert decoded.params[:2] == [0, VIVE_TRACKER]

        room = OscMessageBuilder(address=ROOM_MATRIX_ADDRESS)
        for value in client.rooms[0]:
            room.add_arg(value)
        assert b",ffffffffffff\x00" in room.build().dgram

    def test_indices_follow_role_order_from_zero(self):
        sender = VMTSender(ROLES, client=FakeClient())
        assert sender.slots == {
            TrackerRole.HIP: 0,
            TrackerRole.LEFT_FOOT: 1,
            TrackerRole.RIGHT_FOOT: 2,
        }
        assert device_name(sender.slots[TrackerRole.RIGHT_FOOT]) == "VMT_2"

    def test_mode_defaults_to_vive_tracker_and_can_be_plain(self):
        client = FakeClient()
        VMTSender(ROLES, client=client).send([make_target(TrackerRole.HIP, [0, 1, 0])])
        VMTSender(ROLES, client=client, mode=TRACKER).send(
            [make_target(TrackerRole.HIP, [0, 1, 0])]
        )
        assert [m[1] for m in client.poses] == [VIVE_TRACKER, TRACKER]

    def test_unassigned_roles_are_skipped(self):
        client = FakeClient()
        VMTSender([TrackerRole.HIP], client=client).send(
            [make_target(TrackerRole.HIP, [0, 1, 0]), make_target(TrackerRole.CHEST, [0, 1.4, 0])]
        )
        assert len(client.poses) == 1


class TestSpaces:
    def test_without_a_room_transform_play_space_passes_through(self):
        client = FakeClient()
        rotation = rotation_from_forward_up(np.array([1.0, 0.0, -1.0]))
        VMTSender(ROLES, client=client).send(
            [make_target(TrackerRole.HIP, [0.3, 0.9, -0.4], rotation)]
        )
        _, _, _, position, sent_rotation = unpack(client.poses[0])
        assert np.allclose(position, [0.3, 0.9, -0.4])
        assert np.allclose(sent_rotation, rotation, atol=1e-9)

    def test_identity_rotation_is_the_identity_quaternion(self):
        client = FakeClient()
        VMTSender(ROLES, client=client).send([make_target(TrackerRole.HIP, [0, 1, 0])])
        assert client.poses[0][6:10] == pytest.approx([0.0, 0.0, 0.0, 1.0])

    def test_poses_are_converted_into_raw_space(self):
        """SteamVR maps raw to standing; the sender has to undo exactly that."""
        standing_from_raw = np.hstack([yaw(70.0), np.array([[1.2], [0.05], [-0.7]])])
        client = FakeClient()
        sender = VMTSender(ROLES, client=client, standing_from_raw=lambda: standing_from_raw)
        rotation = rotation_from_forward_up(np.array([0.3, 0.0, -1.0]))
        position = np.array([0.4, 0.95, -0.3])
        sender.send([make_target(TrackerRole.HIP, position, rotation)])

        _, _, _, raw_position, raw_rotation = unpack(client.poses[0])
        assert np.allclose(
            standing_from_raw[:, :3] @ raw_position + standing_from_raw[:, 3], position
        )
        assert np.allclose(standing_from_raw[:, :3] @ raw_rotation, rotation, atol=1e-9)

    def test_waits_until_the_room_transform_is_known(self):
        client = FakeClient()
        room = {"matrix": None}
        sender = VMTSender(ROLES, client=client, standing_from_raw=lambda: room["matrix"])
        assert sender.send([make_target(TrackerRole.HIP, [0, 1, 0])], force=True) is False
        assert client.messages == []

        room["matrix"] = np.hstack([np.eye(3), np.zeros((3, 1))])
        assert sender.send([make_target(TrackerRole.HIP, [0, 1, 0])], force=True) is True
        assert len(client.poses) == 1


class TestRoomMatrix:
    def test_is_set_before_the_first_pose(self):
        client = FakeClient()
        VMTSender(ROLES, client=client).send([make_target(TrackerRole.HIP, [0, 1, 0])])
        assert client.messages[0][0] == ROOM_MATRIX_ADDRESS

    def test_is_standing_to_raw_row_by_row(self):
        """What VMT's manager sends: GetWorkingStandingZeroPoseToRawTrackingPose."""
        standing_from_raw = np.hstack([yaw(-35.0), np.array([[0.6], [0.0], [1.1]])])
        client = FakeClient()
        sender = VMTSender(ROLES, client=client, standing_from_raw=lambda: standing_from_raw)
        sender.send([make_target(TrackerRole.HIP, [0, 1, 0])])

        raw_from_standing = np.array(client.rooms[0]).reshape(3, 4)
        standing = np.array([0.4, 0.95, -0.3])
        raw = standing_from_raw[:, :3].T @ (standing - standing_from_raw[:, 3])
        assert np.allclose(raw_from_standing[:, :3] @ standing + raw_from_standing[:, 3], raw)
        assert all(type(v) is float for v in client.rooms[0])

    def test_is_refreshed_now_and_then_rather_than_every_frame(self):
        client = FakeClient()
        sender = VMTSender(ROLES, client=client, rate_hz=0.0)
        targets = [make_target(TrackerRole.HIP, [0, 1, 0])]
        for frame in range(150):
            sender.send(targets, now=frame / 30.0)
        assert len(client.poses) == 150
        assert 2 <= len(client.rooms) <= 4


class TestLifecycle:
    def test_rate_limit_drops_early_frames(self):
        client = FakeClient()
        sender = VMTSender(ROLES, client=client, rate_hz=60.0)
        targets = [make_target(TrackerRole.HIP, [0, 1, 0])]
        assert sender.send(targets, now=0.0) is True
        assert sender.send(targets, now=0.005) is False
        assert sender.send(targets, now=0.02) is True

    def test_close_switches_every_tracker_off(self):
        client = FakeClient()
        sender = VMTSender(ROLES, client=client)
        sender.send([make_target(role, [0, 1, 0]) for role in ROLES])
        client.messages.clear()
        sender.close()
        assert sorted(m[0] for m in client.poses) == [0, 1, 2]
        assert all(m[1] == DISABLED for m in client.poses)

    def test_close_sends_nothing_if_nothing_was_sent(self):
        client = FakeClient()
        VMTSender(ROLES, client=client).close()
        assert client.messages == []


class TestConfig:
    def test_target_selects_the_sender(self):
        assert isinstance(make_sender(OSCConfig(target="vmt"), None), VMTSender)
        assert isinstance(make_sender(OSCConfig(target="vrchat"), None), VRChatOSCSender)

    def test_vmt_reads_the_room_from_the_vr_source(self):
        class Source:
            standing_from_raw = None

        source = Source()
        sender = make_sender(OSCConfig(target="vmt"), source)
        assert sender._transform() is None
        source.standing_from_raw = np.hstack([np.eye(3), np.ones((3, 1))])
        rotation, translation = sender._transform()
        assert np.allclose(translation, 1.0)

    def test_auto_uses_vmt_only_when_it_is_installed(self, tmp_path, monkeypatch):
        import bodytracker.app.runtime as runtime

        monkeypatch.setattr(runtime, "find_vmt_driver", lambda: tmp_path)
        assert isinstance(make_sender(OSCConfig(target="auto"), None), VMTSender)
        monkeypatch.setattr(runtime, "find_vmt_driver", lambda: None)
        assert isinstance(make_sender(OSCConfig(target="auto"), None), VRChatOSCSender)

    def test_rejects_an_unknown_target(self):
        with pytest.raises(ValueError, match="unknown output target"):
            make_sender(OSCConfig(target="slime"), None)


class TestFindDriver:
    def write_paths(self, tmp_path, drivers):
        openvr = tmp_path / "openvr"
        openvr.mkdir()
        (openvr / "openvrpaths.vrpath").write_text(
            json.dumps({"external_drivers": [str(d) for d in drivers], "runtime": []}),
            encoding="utf-8",
        )

    def test_finds_a_registered_driver(self, tmp_path, monkeypatch):
        driver = tmp_path / "vmt"
        driver.mkdir()
        (driver / "driver.vrdrivermanifest").write_text(json.dumps({"name": "vmt"}), "utf-8")
        other = tmp_path / "other"
        other.mkdir()
        (other / "driver.vrdrivermanifest").write_text(json.dumps({"name": "slimevr"}), "utf-8")
        self.write_paths(tmp_path, [other, driver])
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert find_vmt_driver() == driver

    def test_none_when_not_registered(self, tmp_path, monkeypatch):
        self.write_paths(tmp_path, [tmp_path / "missing"])
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert find_vmt_driver() is None

    def test_none_without_steamvr(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert find_vmt_driver() is None
