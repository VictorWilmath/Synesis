"""Measuring whether the tracker is actually any good.

Held-out anchor error is the centrepiece: hide a SteamVR device from the
lifter, predict where it should be, and compare. It gives true metric error
with no mocap suit, on the user's own hardware.
"""

from .compare import report as compare_report
from .metrics import SessionMetrics, compute, still_mask
from .replay import HeldOutError, held_out_errors, replay, withhold_devices
from .session import Session, SessionMeta, SessionRecorder

__all__ = [
    "HeldOutError",
    "Session",
    "SessionMeta",
    "SessionMetrics",
    "SessionRecorder",
    "compare_report",
    "compute",
    "held_out_errors",
    "replay",
    "still_mask",
    "withhold_devices",
]
