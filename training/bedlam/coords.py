"""Getting BEDLAM's geometry into this project's coordinate conventions.

Four frames are in play and three of them are nearly the same, which is what
makes this dangerous rather than merely tedious.

===================  ==========  ======  =========  =======
Frame                Handedness  Up      Forward    Units
===================  ==========  ======  =========  =======
Unreal world         left        +z      +x         cm
BEDLAM "world"       right       -y      +z         m
OpenCV camera        right       -y      +z         m
Play space (ours)    right       +y      -z         m
===================  ==========  ======  =========  =======

BEDLAM's "world" deserves the scare quotes. It is not a scene-global frame: the
camera's yaw is folded into each body's global orientation, so it is a
per-sequence frame that is yaw-aligned to the camera. That is harmless here,
because the yaw relationship between a webcam and a VR play space is arbitrary
anyway and gets solved by calibration at runtime. It would not be harmless if
you were trying to place two cameras in one scene.

Everything below follows ``data_processing/df_full_body.py`` in the BEDLAM
repository, which is the only authoritative statement of these conventions.
Where a constant looks arbitrary it is because it is: BEDLAM's own code has it
too, and matching it exactly matters more than it looking principled.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The virtual filmback every BEDLAM render uses, in millimetres. Landscape by
# default; the rotated scenes swap the two.
SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 20.25
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# BEDLAM "world" is X right, Y down, Z forward. Play space is X right, Y up,
# Z backward. Flipping two axes preserves handedness; flipping one would
# mirror the skeleton, which is the classic way to end up with a left-handed
# avatar and no idea why.
WORLD_FROM_BEDLAM = np.diag([1.0, -1.0, -1.0])


def unreal_to_opencv(points: np.ndarray) -> np.ndarray:
    """Unreal's (x forward, y right, z up) to OpenCV's (x right, y down, z forward).

    Exactly BEDLAM's ``unreal2cv2``: roll the axes then negate the new y.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    rolled = np.roll(pts, 2, axis=1)
    out = rolled * np.array([1.0, -1.0, 1.0])
    return out.reshape(np.shape(points))


def focal_mm_to_px(focal_mm: float, sensor_mm: float, image_px: int) -> float:
    """A focal length on a physical filmback, in pixels."""
    return float(focal_mm) * float(image_px) / float(sensor_mm)


def intrinsics(
    focal_mm: float,
    *,
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    sensor_width_mm: float = SENSOR_WIDTH_MM,
    sensor_height_mm: float = SENSOR_HEIGHT_MM,
) -> np.ndarray:
    """The 3x3 camera matrix for one frame.

    Per frame, not per sequence: BEDLAM 2.0 includes dolly-zoom shots where the
    focal length changes mid-sequence, which is why the CSV carries it on every
    row.

    The principal point is exactly the image centre and the pixels come out
    square, since both axes divide the same filmback.
    """
    return np.array(
        [
            [focal_mm_to_px(focal_mm, sensor_width_mm, width), 0.0, width / 2.0],
            [0.0, focal_mm_to_px(focal_mm, sensor_height_mm, height), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """Rotation matrix from an axis-angle vector, without pulling in OpenCV."""
    vector = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3)
    axis = vector / theta
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def camera_rotation(pitch_deg: float, roll_deg: float, yaw_deg: float = 0.0) -> np.ndarray:
    """Rotation from BEDLAM's world frame into camera orientation.

    Unreal's rotator is applied yaw, then pitch, then roll, and once the axes
    are in OpenCV order those become rotations about y, x and z respectively.
    All three angles are negated relative to the CSV, which is what BEDLAM's
    own reader does and follows from Unreal's yaw being left-handed while its
    pitch and roll are right-handed.

    Yaw defaults to zero because the frame this maps out of is already
    yaw-aligned to the camera. Pass it only if you are working from the raw
    Unreal CSV and want a genuinely scene-global frame.
    """
    yaw = _rodrigues([0.0, np.radians(-yaw_deg), 0.0])
    pitch = _rodrigues([np.radians(-pitch_deg), 0.0, 0.0])
    roll = _rodrigues([0.0, 0.0, np.radians(-roll_deg)])
    return roll @ pitch @ yaw


def rotation_cw_from_extrinsics(cam_ext: np.ndarray) -> np.ndarray:
    """Our world-to-camera rotation, from BEDLAM's 4x4 ``cam_ext``.

    BEDLAM's rotation maps its own down-is-positive world into the camera. Ours
    maps a y-up play space, so the two differ by the axis flip.
    """
    rotation = np.asarray(cam_ext, dtype=np.float64)[:3, :3]
    return rotation @ WORLD_FROM_BEDLAM


def up_in_camera(cam_ext: np.ndarray) -> np.ndarray:
    """Which way is up, seen from the camera.

    The single most useful thing the extrinsics give the lifter: it is what
    lets gravity break the depth ambiguity when nothing else can.
    """
    return rotation_cw_from_extrinsics(cam_ext) @ np.array([0.0, 1.0, 0.0])


def project(points_camera: np.ndarray, cam_int: np.ndarray) -> np.ndarray:
    """Pinhole-project camera-space metres to pixels.

    No distortion, because BEDLAM renders through an ideal pinhole. Real lens
    distortion is added later by the degradation pipeline, where it belongs.
    """
    pts = np.atleast_2d(np.asarray(points_camera, dtype=np.float64))
    depth = pts[:, 2:3]
    normalised = np.divide(
        pts, depth, out=np.zeros_like(pts), where=np.abs(depth) > 1e-9
    )
    normalised[:, 2] = 1.0
    return normalised @ np.asarray(cam_int, dtype=np.float64).T


def behind_camera(points_camera: np.ndarray, margin_m: float = 0.1) -> np.ndarray:
    """Which points are behind the lens, and so project to nonsense.

    Worth checking explicitly. A point just behind the camera projects to a
    plausible-looking pixel on the far side of the image, so it survives any
    bounds check and quietly poisons the training pair.
    """
    pts = np.atleast_2d(np.asarray(points_camera, dtype=np.float64))
    return pts[:, 2] < margin_m


@dataclass(slots=True)
class CameraFrame:
    """One row of a BEDLAM camera CSV, still in Unreal's units and axes."""

    name: str
    position_cm: np.ndarray
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    focal_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    hfov_deg: float

    @property
    def position_m(self) -> np.ndarray:
        """Camera position in OpenCV axes, metres."""
        return unreal_to_opencv(self.position_cm.reshape(1, 3) / 100.0).reshape(3)


def read_camera_csv(path: Path | str) -> list[CameraFrame]:
    """Read ``ground_truth/camera/<seq>_camera.csv``, one row per rendered frame.

    Columns are ``name,x,y,z,yaw,pitch,roll,focal_length,sensor_width,
    sensor_height,hfov``. BEDLAM 2.0 writes the same columns from EXR metadata
    instead of Blueprint logging, so the same reader serves both.
    """
    rows: list[CameraFrame] = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                CameraFrame(
                    name=row["name"],
                    position_cm=np.array(
                        [float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64
                    ),
                    yaw_deg=float(row["yaw"]),
                    pitch_deg=float(row["pitch"]),
                    roll_deg=float(row["roll"]),
                    focal_mm=float(row["focal_length"]),
                    sensor_width_mm=float(row.get("sensor_width", SENSOR_WIDTH_MM)),
                    sensor_height_mm=float(row.get("sensor_height", SENSOR_HEIGHT_MM)),
                    hfov_deg=float(row.get("hfov", 0.0)),
                )
            )
    return rows


def camera_is_static(frames: list[CameraFrame], tolerance_m: float = 0.01) -> bool:
    """Whether the camera really held still, rather than merely being named static.

    Checked rather than trusted. BEDLAM 2.0 layers Perlin camera shake over
    some shots, and a sequence whose extrinsics wander breaks the assumption
    that one calibration holds for the whole clip.
    """
    if len(frames) < 2:
        return True
    positions = np.stack([f.position_m for f in frames])
    angles = np.stack([[f.yaw_deg, f.pitch_deg, f.roll_deg] for f in frames])
    moved = float(np.ptp(positions, axis=0).max())
    turned = float(np.ptp(angles, axis=0).max())
    return moved <= tolerance_m and turned <= 0.5


def camera_height(floor_points_camera: np.ndarray, cam_ext: np.ndarray) -> float:
    """How far the camera is above the floor, in metres.

    Measured from points known to be on the floor, which in practice means the
    lowest foot over a sequence. The camera never sees the floor plane
    directly, so a body standing on it is the only ruler available, which is
    also exactly how the runtime finds the floor.
    """
    up = up_in_camera(cam_ext)
    points = np.atleast_2d(np.asarray(floor_points_camera, dtype=np.float64))
    # Height of each point along world up, relative to the camera. The floor is
    # below, so these are negative; the camera's height is the least negative
    # one negated, using the highest "floor" point to resist a foot that has
    # sunk through the ground in the render.
    heights = points @ up
    return float(-np.max(heights))


def play_space_transform(
    cam_ext: np.ndarray,
    camera_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the world-to-camera extrinsics the runtime pipeline expects.

    BEDLAM's world frame is centred on whichever body is being described, which
    is no use as a room frame. This puts the origin where a real setup puts it:
    on the floor, directly below the camera, y up. The model then sees the same
    frame at training time that it will be asked about at runtime, where the
    origin comes from SteamVR's floor calibration.

    Returns ``(rotation_cw, translation_cw)`` such that
    ``x_camera = rotation_cw @ x_play + translation_cw``.
    """
    rotation_cw = rotation_cw_from_extrinsics(cam_ext)
    # Placing the origin directly below the camera means the camera itself ends
    # up at (0, height, 0) in play space, which is the whole point.
    translation_cw = -camera_height_m * (rotation_cw @ np.array([0.0, 1.0, 0.0]))
    return rotation_cw, translation_cw
