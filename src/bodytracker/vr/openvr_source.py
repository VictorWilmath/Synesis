"""Reading head and controller poses from SteamVR via pyopenvr.

Two deliberate choices:

*Background application type.* We do not render, and we should not be the
reason SteamVR launches. Initialising as a background app means the tracker
attaches to a running SteamVR and fails cleanly if there is none.

*Polling rather than ``waitGetPoses``.* The compositor's wait function blocks
until the next vsync and is meant for applications that submit frames. This
loop is driven by the camera at 30-60 Hz, not by the display, so it asks for
poses directly and never blocks.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from ..geom import play_from_steamvr_matrix
from ..types import DevicePose, VRState

log = logging.getLogger(__name__)


def _matrix_to_numpy(matrix) -> np.ndarray:
    """Convert an ``HmdMatrix34_t`` to a (3, 4) array."""
    return np.array([[matrix[i][j] for j in range(4)] for i in range(3)], dtype=np.float64)


class OpenVRSource:
    """Head and hand poses from SteamVR, in play space."""

    def __init__(self, prediction_seconds: float = 0.0) -> None:
        # Poses are consumed alongside a camera frame that is already tens of
        # milliseconds old, so predicting forward would desynchronise them.
        self.prediction_seconds = prediction_seconds
        self._openvr = None
        self._system = None
        self._device_count = 0

    def open(self) -> None:
        import openvr

        self._openvr = openvr
        try:
            openvr.init(openvr.VRApplication_Background)
        except Exception as exc:
            raise RuntimeError(
                f"could not attach to SteamVR. Is it running? ({type(exc).__name__}: {exc})"
            ) from exc

        self._system = openvr.VRSystem()
        self._device_count = openvr.k_unMaxTrackedDeviceCount
        log.info("attached to SteamVR")

    def close(self) -> None:
        if self._openvr is not None:
            try:
                self._openvr.shutdown()
            except Exception:
                log.debug("openvr shutdown raised", exc_info=True)
            self._openvr = None
            self._system = None

    def _pose_at(self, poses, index: int, timestamp: float) -> DevicePose | None:
        if index < 0 or index >= len(poses):
            return None
        pose = poses[index]
        if not pose.bPoseIsValid or not pose.bDeviceIsConnected:
            return None
        position, rotation = play_from_steamvr_matrix(
            _matrix_to_numpy(pose.mDeviceToAbsoluteTracking)
        )
        return DevicePose(position=position, rotation=rotation, valid=True, timestamp=timestamp)

    def _hand_indices(self) -> tuple[int, int]:
        """Device indices for the left and right controllers, or -1 if absent."""
        openvr = self._openvr
        left = self._system.getTrackedDeviceIndexForControllerRole(
            openvr.TrackedControllerRole_LeftHand
        )
        right = self._system.getTrackedDeviceIndexForControllerRole(
            openvr.TrackedControllerRole_RightHand
        )
        invalid = openvr.k_unTrackedDeviceIndexInvalid
        return (-1 if left == invalid else left, -1 if right == invalid else right)

    def poll(self) -> VRState | None:
        if self._system is None:
            return None
        openvr = self._openvr
        timestamp = time.monotonic()

        poses = self._system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            self.prediction_seconds,
            self._device_count,
        )

        left_index, right_index = self._hand_indices()
        return VRState(
            head=self._pose_at(poses, openvr.k_unTrackedDeviceIndex_Hmd, timestamp),
            left_hand=self._pose_at(poses, left_index, timestamp),
            right_hand=self._pose_at(poses, right_index, timestamp),
            timestamp=timestamp,
        )

    def __enter__(self) -> OpenVRSource:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
