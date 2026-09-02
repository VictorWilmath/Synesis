"""Coordinate spaces and rotation math.

Nothing outside this package should be flipping signs or reordering axes by
hand. Every conversion between the camera, SteamVR and Unity conventions lives
here and is unit-tested.
"""

from .rotations import (
    euler_zxy_degrees_from_matrix,
    geodesic_angle_degrees,
    look_rotation,
    matrix_from_euler_zxy_degrees,
    matrix_from_quaternion,
    orthonormalize,
    quaternion_from_matrix,
    rotation_from_forward_up,
    slerp,
)
from .spaces import (
    camera_from_play,
    intrinsics_from_fov,
    look_at_extrinsics,
    play_from_camera,
    play_from_steamvr_matrix,
    project_points,
    unity_position_from_play,
    unity_rotation_from_play,
)

__all__ = [
    "camera_from_play",
    "euler_zxy_degrees_from_matrix",
    "geodesic_angle_degrees",
    "intrinsics_from_fov",
    "look_at_extrinsics",
    "look_rotation",
    "matrix_from_euler_zxy_degrees",
    "matrix_from_quaternion",
    "orthonormalize",
    "play_from_camera",
    "play_from_steamvr_matrix",
    "project_points",
    "quaternion_from_matrix",
    "rotation_from_forward_up",
    "slerp",
    "unity_position_from_play",
    "unity_rotation_from_play",
]
