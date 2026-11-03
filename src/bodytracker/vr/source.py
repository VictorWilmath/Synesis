"""The VR pose source interface, plus a playback implementation.

Keeping this behind an interface means the pipeline, the calibrator and the
eval harness can all be driven from a recording with no headset attached, which
matters because the interesting failures are the ones you want to replay.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..types import VRState
from .history import interpolate_state


class VRSource(Protocol):
    """Supplies head and hand poses in play space."""

    def open(self) -> None: ...

    def poll(self) -> VRState | None: ...

    def close(self) -> None: ...


class RecordedVRSource:
    """Replays captured VR states, matched to the nearest timestamp.

    Camera frames and SteamVR poses arrive on unrelated clocks, so playback has
    to interpolate rather than assume they line up.
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
        return interpolate_state(self.states, self._times, timestamp)

    def at_many(self, timestamps: Sequence[float]) -> list[VRState | None]:
        return [self.at(t) for t in timestamps]


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
    return standing_height_from_head_heights(
        [h for h in (head_height(s) for s in states) if h is not None]
    )


def standing_height_from_head_heights(heights: Sequence[float]) -> float | None:
    """Standing height from raw headset heights above the floor."""
    usable = [h for h in heights if h > 0.5]
    if len(usable) < 10:
        return None
    # The HMD sits roughly at eye level, which is about 0.93 of stature.
    return float(np.percentile(usable, 90) / 0.93)
