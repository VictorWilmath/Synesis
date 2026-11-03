"""SteamVR device poses, the metric anchor this whole approach rests on."""

from .history import VRHistory, interpolate_state
from .source import RecordedVRSource, VRSource

__all__ = ["RecordedVRSource", "VRHistory", "VRSource", "interpolate_state"]


def __getattr__(name: str):
    # Deferred so importing this package does not require the openvr binding.
    if name == "OpenVRSource":
        from .openvr_source import OpenVRSource

        return OpenVRSource
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
