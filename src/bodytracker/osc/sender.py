"""Sends tracker poses to VRChat over OSC.

VRChat's contract, from its OSC trackers documentation:

- UDP to port 9000
- ``/tracking/trackers/{1..8}/position`` with three floats, world-space metres
- ``/tracking/trackers/{1..8}/rotation`` with three floats, **Euler degrees in
  ZXY order** (Unity's convention, not a quaternion)
- ``/tracking/trackers/head/{position,rotation}`` as an optional alignment
  reference
- There is no tracker 0

Two details that are easy to get wrong and painful to debug from inside VRChat:

*Slot assignment must be fixed for the whole session.* VRChat's full-body
calibration binds each numbered slot to a body point, so reordering slots
mid-session silently swaps the user's limbs.

*Never drop a tracker mid-session.* A slot that stops updating is better than
one that jumps to the origin, so stale values are still transmitted; see
`filters.hold.HoldLastValid`.

Because this pipeline works natively in SteamVR play space, which is also
VRChat's tracking space, the head alignment endpoints are optional. They exist
for senders that produce poses in some arbitrary space of their own.
"""

from __future__ import annotations

import time

import numpy as np
from pythonosc import udp_client

from ..geom import (
    euler_zxy_degrees_from_matrix,
    unity_position_from_play,
    unity_rotation_from_play,
)
from ..skeleton import MAX_TRACKERS, TrackerRole, assign_slots
from ..types import DevicePose, TrackerTarget

HEAD_POSITION_ADDRESS = "/tracking/trackers/head/position"
HEAD_ROTATION_ADDRESS = "/tracking/trackers/head/rotation"


def position_address(slot: int) -> str:
    if not 1 <= slot <= MAX_TRACKERS:
        raise ValueError(f"tracker slot must be 1..{MAX_TRACKERS}, got {slot}")
    return f"/tracking/trackers/{slot}/position"


def rotation_address(slot: int) -> str:
    if not 1 <= slot <= MAX_TRACKERS:
        raise ValueError(f"tracker slot must be 1..{MAX_TRACKERS}, got {slot}")
    return f"/tracking/trackers/{slot}/rotation"


class VRChatOSCSender:
    """Converts play-space tracker poses to VRChat's wire format and sends them.

    Set `client` to inject a fake in tests; anything with a
    ``send_message(address, value)`` method works.
    """

    def __init__(
        self,
        roles: list[TrackerRole],
        *,
        host: str = "127.0.0.1",
        port: int = 9000,
        send_head: bool = False,
        rate_hz: float = 60.0,
        client: object | None = None,
    ) -> None:
        self.slots = assign_slots(roles)
        self.roles = list(roles)
        self.send_head = send_head
        self.min_interval = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self._client = client or udp_client.SimpleUDPClient(host, port)
        # None rather than 0.0 so the very first frame is never rate-limited.
        self._last_send: float | None = None
        self.frames_sent = 0
        self.messages_sent = 0

    # -- wire format ------------------------------------------------------

    @staticmethod
    def _to_wire(position: np.ndarray, rotation: np.ndarray) -> tuple[list[float], list[float]]:
        """Play space to VRChat's expected floats."""
        unity_pos = unity_position_from_play(position)
        unity_rot = unity_rotation_from_play(rotation)
        euler = euler_zxy_degrees_from_matrix(unity_rot)
        return [float(v) for v in unity_pos], [float(v) for v in euler]

    def _emit(self, address: str, values: list[float]) -> None:
        self._client.send_message(address, values)
        self.messages_sent += 1

    # -- public API -------------------------------------------------------

    def send(
        self,
        targets: list[TrackerTarget],
        head: DevicePose | None = None,
        *,
        now: float | None = None,
        force: bool = False,
    ) -> bool:
        """Transmit one frame. Returns False if skipped by the rate limiter."""
        timestamp = time.monotonic() if now is None else now
        if (
            not force
            and self.min_interval > 0
            and self._last_send is not None
            and timestamp - self._last_send < self.min_interval
        ):
            return False
        self._last_send = timestamp

        for target in targets:
            slot = self.slots.get(target.role)
            if slot is None:
                # Role is being estimated but the user chose not to send it.
                continue
            position, euler = self._to_wire(target.position, target.rotation)
            self._emit(position_address(slot), position)
            self._emit(rotation_address(slot), euler)

        if self.send_head and head is not None and head.valid:
            position, euler = self._to_wire(head.position, head.rotation)
            self._emit(HEAD_POSITION_ADDRESS, position)
            self._emit(HEAD_ROTATION_ADDRESS, euler)

        self.frames_sent += 1
        return True

    def close(self) -> None:
        transport = getattr(self._client, "_sock", None)
        if transport is not None:
            transport.close()

    def __enter__(self) -> VRChatOSCSender:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
