"""The VR pose source interface, plus a playback implementation.

Keeping this behind an interface means the pipeline, the calibrator and the
eval harness can all be driven from a recording with no headset attached, which
matters because the interesting failures are the ones you want to replay.
"""

from __future__ import annotations

import bisect
from typing import Protocol

import numpy as np

from ..types import DevicePose, VRState


class VRSource(Protocol):
    """Supplies head and hand poses in play space."""

    def open(self) -> None: ...

    def poll(self) -> VRState | None: ...

    def close(self) -> None: ...


class RecordedVRSource:
    """Replays captured VR states, matched to the nearest timestamp.

    Camera frames and SteamVR poses arrive on unrelated clocks, so playback has
    to interpolate rather than assume they line up. Positions are interpolated
    linearly; rotations take the nearest sample, which is accurate enough at
    90 Hz.
    """

    def __init__(self, states: list[VRState]) -> None:
        self.states = sorted(states, key=lambda s: s.timestamp)
        self._times = [s.timestamp for s in self.states]
        self._cursor = 0

    def open(self) -> None:
        self._cursor = 0

    def close(self) -> None:
        pass

    def poll(self) -> VRState | None:
        """Return the next state in sequence, for stepping through a recording."""
        if self._cursor >= len(self.states):
            return None
        state = self.states[self._cursor]
        self._cursor += 1
        return state

    def at(self, timestamp: float) -> VRState | None:
        """The VR state at an arbitrary time, interpolated between samples."""
        if not self.states:
            return None
        if timestamp <= self._times[0]:
            return self.states[0]
        if timestamp >= self._times[-1]:
            return self.states[-1]

        index = bisect.bisect_left(self._times, timestamp)
        before, after = self.states[index - 1], self.states[index]
        span = after.timestamp - before.timestamp
        if span <= 0:
            return after
        weight = (timestamp - before.timestamp) / span

        return VRState(
            head=_lerp_pose(before.head, after.head, weight, timestamp),
            left_hand=_lerp_pose(before.left_hand, after.left_hand, weight, timestamp),
            right_hand=_lerp_pose(before.right_hand, after.right_hand, weight, timestamp),
            timestamp=timestamp,
        )


def _lerp_pose(
    before: DevicePose | None,
    after: DevicePose | None,
    weight: float,
    timestamp: float,
) -> DevicePose | None:
    if before is None or after is None or not before.valid or not after.valid:
        return after if after is not None and after.valid else before
    return DevicePose(
        position=(1.0 - weight) * before.position + weight * after.position,
        rotation=(after if weight >= 0.5 else before).rotation.copy(),
        valid=True,
        timestamp=timestamp,
    )


def head_height(state: VRState) -> float | None:
    """Height of the headset above the floor, useful for a standing-height check."""
    if state.head is None or not state.head.valid:
        return None
    return float(state.head.position[1])


def estimate_standing_height(states: list[VRState]) -> float | None:
    """Infer standing height from the headset over a recording.

    Uses a high percentile of head height rather than the maximum, so a jump or
    a moment of reaching up does not set the value, then adds the offset from
    the headset to the top of the head.
    """
    heights = [h for h in (head_height(s) for s in states) if h is not None and h > 0.5]
    if len(heights) < 10:
        return None
    # The HMD sits roughly at eye level, which is about 0.93 of stature.
    return float(np.percentile(heights, 90) / 0.93)
