"""Sends tracker poses to SteamVR as virtual trackers, through Virtual Motion Tracker.

VRChat's OSC trackers are a second, separate way into its full-body system,
with their own idea of where their space sits. Real trackers have none of
that: SteamVR places them in the same space as the headset, VRChat calibrates
against them with its normal full-body flow, and SteamVR itself shows where
they are. Emulating real trackers is how Virtual Desktop's body tracking gets
into VRChat.

Virtual Motion Tracker (VMT, https://github.com/gpsnmeajp/VirtualMotionTracker)
is a SteamVR driver that does the emulating. It creates up to 58 trackers and
moves each to whatever pose arrives over OSC on UDP 39570.

Poses go to ``/VMT/Raw/Driver``: OpenVR's right-handed axes, a quaternion, in
SteamVR's raw driver space. That is not the standing space Synesis works in;
the two differ by the room setup, which SteamVR reports and which changes
whenever the user recentres. Converting here, rather than using VMT's room
space, means recentring cannot leave the trackers behind.

VMT still refuses every pose, raw ones included, until it has been given a
room matrix, which on a fresh install it has not. So the sender also keeps
VMT's temporary room matrix set to the current room, the same matrix VMT's
manager would save, without touching VMT's settings file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pythonosc import udp_client

from ..geom import quaternion_from_matrix
from ..skeleton import TrackerRole, assign_slots
from ..types import DevicePose, TrackerTarget

log = logging.getLogger(__name__)

VMT_PORT = 39570
RAW_DRIVER_ADDRESS = "/VMT/Raw/Driver"
ROOM_MATRIX_ADDRESS = "/VMT/SetRoomMatrix/Temporary"
# Often enough that a restarted SteamVR or VMT is back within moments.
_ROOM_RESEND_S = 2.0

# VMT's device modes. The type is fixed the first time an index is used after
# SteamVR starts, so changing it needs a SteamVR restart.
DISABLED = 0
TRACKER = 1
VIVE_TRACKER = 7


def device_name(index: int) -> str:
    """How the tracker appears in SteamVR, e.g. in Manage Trackers."""
    return f"VMT_{index}"


class VMTSender:
    """Converts play-space tracker poses to VMT messages and sends them.

    `standing_from_raw` returns SteamVR's (3, 4) raw-to-standing transform, or
    None while it is not known yet; frames are skipped until it is. Leave it
    out to treat play space as raw space, which is only right without SteamVR.

    Set `client` to inject a fake in tests; anything with a
    ``send_message(address, value)`` method works.
    """

    def __init__(
        self,
        roles: list[TrackerRole],
        *,
        host: str = "127.0.0.1",
        port: int = VMT_PORT,
        mode: int = VIVE_TRACKER,
        rate_hz: float = 60.0,
        standing_from_raw: Callable[[], np.ndarray | None] | None = None,
        client: object | None = None,
    ) -> None:
        # The same fixed role order as VRChat's slots, numbered from zero.
        self.slots = {role: slot - 1 for role, slot in assign_slots(roles).items()}
        self.roles = list(roles)
        self.mode = mode
        self.min_interval = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self._standing_from_raw = standing_from_raw
        self._client = client or udp_client.SimpleUDPClient(host, port)
        self._last_send: float | None = None
        self._room_sent_at: float | None = None
        self.frames_sent = 0
        self.messages_sent = 0

    def _transform(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._standing_from_raw is None:
            return np.eye(3), np.zeros(3)
        matrix = self._standing_from_raw()
        if matrix is None:
            return None
        matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 4)
        return matrix[:, :3], matrix[:, 3]

    @staticmethod
    def _to_wire(
        position: np.ndarray,
        rotation: np.ndarray,
        standing_rotation: np.ndarray,
        standing_translation: np.ndarray,
    ) -> list[float]:
        """Play space to raw space: position then quaternion as x, y, z, w."""
        raw_position = standing_rotation.T @ (
            np.asarray(position, dtype=np.float64) - standing_translation
        )
        raw_rotation = standing_rotation.T @ np.asarray(rotation, dtype=np.float64)
        w, x, y, z = quaternion_from_matrix(raw_rotation)
        return [float(v) for v in raw_position] + [float(x), float(y), float(z), float(w)]

    def _emit(self, index: int, mode: int, pose: list[float]) -> None:
        self._client.send_message(RAW_DRIVER_ADDRESS, [int(index), int(mode), 0.0, *pose])
        self.messages_sent += 1

    def _set_room(self, standing_rotation: np.ndarray, standing_translation: np.ndarray) -> None:
        """Give VMT the standing-to-raw matrix, row by row, as its manager does."""
        raw_from_standing = np.hstack(
            [standing_rotation.T, (-standing_rotation.T @ standing_translation).reshape(3, 1)]
        )
        self._client.send_message(
            ROOM_MATRIX_ADDRESS, [float(v) for v in raw_from_standing.reshape(-1)]
        )
        self.messages_sent += 1

    def send(
        self,
        targets: list[TrackerTarget],
        head: DevicePose | None = None,
        *,
        now: float | None = None,
        force: bool = False,
    ) -> bool:
        """Transmit one frame. Returns False if skipped.

        `head` is accepted for parity with the VRChat sender and not used:
        SteamVR already knows where the headset is.
        """
        timestamp = time.monotonic() if now is None else now
        if (
            not force
            and self.min_interval > 0
            and self._last_send is not None
            and timestamp - self._last_send < self.min_interval
        ):
            return False
        transform = self._transform()
        if transform is None:
            return False
        self._last_send = timestamp

        rotation, translation = transform
        if self._room_sent_at is None or timestamp - self._room_sent_at >= _ROOM_RESEND_S:
            self._set_room(rotation, translation)
            self._room_sent_at = timestamp
        for target in targets:
            index = self.slots.get(target.role)
            if index is None:
                continue
            self._emit(
                index,
                self.mode,
                self._to_wire(target.position, target.rotation, rotation, translation),
            )

        self.frames_sent += 1
        return True

    def close(self) -> None:
        # Switch the trackers off. Left on, they would sit frozen wherever they
        # were last, and VRChat would keep the avatar's legs there.
        if self.frames_sent:
            for index in self.slots.values():
                try:
                    self._emit(index, DISABLED, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                except OSError:
                    log.debug("could not switch VMT tracker %d off", index, exc_info=True)
        transport = getattr(self._client, "_sock", None)
        if transport is not None:
            transport.close()

    def __enter__(self) -> VMTSender:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def find_vmt_driver() -> Path | None:
    """Where SteamVR has VMT registered, or None if it does not appear to be installed."""
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    try:
        paths = json.loads((Path(local) / "openvr" / "openvrpaths.vrpath").read_text("utf-8"))
    except (OSError, ValueError):
        return None

    candidates = [Path(entry) for entry in paths.get("external_drivers") or []]
    candidates += [Path(entry) / "drivers" / "vmt" for entry in paths.get("runtime") or []]
    for folder in candidates:
        try:
            manifest = json.loads((folder / "driver.vrdrivermanifest").read_text("utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("name") == "vmt":
            return folder
    return None
