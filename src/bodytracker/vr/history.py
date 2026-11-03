"""A short rolling history of SteamVR poses, so camera frames can be matched in time.

A webcam frame reaches Python tens of milliseconds after its exposure, while a
SteamVR pose is current to the millisecond. Pairing a frame with "the latest
headset pose" therefore pairs it with where the user will be, not where they
were. At walking speed that is several centimetres, and at the hands it is
far more, which is easily the largest error in any camera-to-headset fusion.

Polling SteamVR on its own thread and keeping the last couple of seconds lets
the pipeline ask for the pose at the moment the frame was really captured.
"""

from __future__ import annotations

import bisect
import logging
import threading
import time
from collections import deque
from collections.abc import Sequence

from ..geom import slerp
from ..types import DevicePose, VRState

log = logging.getLogger(__name__)


def _interpolate_pose(
    before: DevicePose | None,
    after: DevicePose | None,
    weight: float,
    timestamp: float,
) -> DevicePose | None:
    if before is None or after is None or not before.valid or not after.valid:
        return after if after is not None and after.valid else before
    return DevicePose(
        position=(1.0 - weight) * before.position + weight * after.position,
        rotation=slerp(before.rotation, after.rotation, weight),
        valid=True,
        timestamp=timestamp,
    )


def interpolate_state(
    states: Sequence[VRState],
    times: Sequence[float],
    timestamp: float,
    *,
    tolerance_s: float | None = None,
) -> VRState | None:
    """The VR state at `timestamp`, interpolated between the bracketing samples.

    With `tolerance_s` set, a request further than that outside the recorded
    span returns None instead of the nearest end. A pose from a quarter of a
    second away is not the pose at this frame, and a calibrator that silently
    accepts it learns the wrong camera.
    """
    if not states:
        return None
    if timestamp <= times[0]:
        if tolerance_s is not None and times[0] - timestamp > tolerance_s:
            return None
        return states[0]
    if timestamp >= times[-1]:
        if tolerance_s is not None and timestamp - times[-1] > tolerance_s:
            return None
        return states[-1]

    index = bisect.bisect_left(times, timestamp)
    before, after = states[index - 1], states[index]
    span = after.timestamp - before.timestamp
    if span <= 0:
        return after
    weight = (timestamp - before.timestamp) / span
    return VRState(
        head=_interpolate_pose(before.head, after.head, weight, timestamp),
        left_hand=_interpolate_pose(before.left_hand, after.left_hand, weight, timestamp),
        right_hand=_interpolate_pose(before.right_hand, after.right_hand, weight, timestamp),
        timestamp=timestamp,
    )


class VRHistory:
    """Polls a VR source in the background and answers "where was everything at t"."""

    def __init__(
        self,
        source: object | None,
        *,
        poll_hz: float = 90.0,
        span_s: float = 2.0,
        tolerance_s: float = 0.05,
    ) -> None:
        self.source = source
        self.interval_s = 1.0 / poll_hz if poll_hz > 0 else 0.0
        self.tolerance_s = tolerance_s
        self._states: deque[VRState] = deque(maxlen=max(2, int(poll_hz * span_s) + 1))
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.poll_failures = 0

    def start(self) -> None:
        if self.source is None or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vr-history", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        next_poll = time.monotonic()
        while not self._stop.is_set():
            try:
                state = self.source.poll()
            except Exception:
                self.poll_failures += 1
                if self.poll_failures == 1:
                    log.exception("SteamVR poll failed; retrying")
                state = None
            if state is not None:
                self.record(state)
            next_poll += self.interval_s
            self._stop.wait(max(0.0, next_poll - time.monotonic()))
            if next_poll < time.monotonic() - self.interval_s:
                next_poll = time.monotonic()

    def record(self, state: VRState) -> None:
        with self._lock:
            if self._states and state.timestamp <= self._states[-1].timestamp:
                return
            self._states.append(state)

    def latest(self) -> VRState | None:
        with self._lock:
            return self._states[-1] if self._states else None

    def at(self, timestamp: float) -> VRState | None:
        return self.at_many([timestamp])[0]

    def at_many(self, timestamps: Sequence[float]) -> list[VRState | None]:
        """Several lookups against one consistent snapshot of the history."""
        with self._lock:
            states = list(self._states)
        times = [state.timestamp for state in states]
        return [
            interpolate_state(states, times, t, tolerance_s=self.tolerance_s) for t in timestamps
        ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)
