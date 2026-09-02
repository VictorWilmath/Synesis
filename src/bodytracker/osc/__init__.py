"""VRChat OSC tracker output."""

from .sender import (
    HEAD_POSITION_ADDRESS,
    HEAD_ROTATION_ADDRESS,
    VRChatOSCSender,
    position_address,
    rotation_address,
)

__all__ = [
    "HEAD_POSITION_ADDRESS",
    "HEAD_ROTATION_ADDRESS",
    "VRChatOSCSender",
    "position_address",
    "rotation_address",
]
