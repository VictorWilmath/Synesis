"""Debug overlay.

Shows the 2D keypoints, the reprojected 3D solution and a live stats readout.
The reprojection is the useful part: when the lifted skeleton stops sitting on
top of the detections, the depth solve has gone wrong, and that is visible here
long before it is obvious in VRChat.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..geom import camera_from_play, project_points
from ..skeleton import KEYPOINT_NAMES, SKELETON_EDGES
from ..types import FrameResult

_GREEN = (0, 220, 0)
_AMBER = (0, 190, 255)
_RED = (0, 0, 255)
_BLUE = (255, 180, 0)
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_keypoints2d(
    frame: np.ndarray,
    xy: np.ndarray,
    scores: np.ndarray,
    min_score: float,
) -> None:
    for a, b in SKELETON_EDGES:
        if scores[a] >= min_score and scores[b] >= min_score:
            cv2.line(
                frame,
                tuple(np.int32(xy[a])),
                tuple(np.int32(xy[b])),
                _GREEN,
                2,
                cv2.LINE_AA,
            )
    for i, point in enumerate(xy):
        colour = _GREEN if scores[i] >= min_score else _RED
        cv2.circle(frame, tuple(np.int32(point)), 3, colour, -1, cv2.LINE_AA)


def draw_reprojection(
    frame: np.ndarray,
    xyz_play: np.ndarray,
    intrinsics: np.ndarray,
    rotation_cw: np.ndarray,
    translation_cw: np.ndarray,
) -> None:
    """Project the 3D solution back into the image.

    If this does not land on the green 2D skeleton, either the extrinsics or
    the depth solve is wrong.
    """
    camera = camera_from_play(xyz_play, rotation_cw, translation_cw)
    if np.any(camera[:, 2] <= 0):
        return
    pixels = project_points(camera, intrinsics)
    for a, b in SKELETON_EDGES:
        cv2.line(
            frame,
            tuple(np.int32(pixels[a])),
            tuple(np.int32(pixels[b])),
            _BLUE,
            1,
            cv2.LINE_AA,
        )


def draw_bbox(frame: np.ndarray, bbox: np.ndarray) -> None:
    x1, y1, x2, y2 = np.int32(bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), _AMBER, 1)


def draw_stats(frame: np.ndarray, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        y = 22 + i * 20
        # Outline first so the text stays readable over a bright background.
        cv2.putText(frame, line, (10, y), _FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), _FONT, 0.5, _WHITE, 1, cv2.LINE_AA)


def render(
    frame: np.ndarray,
    result: FrameResult,
    *,
    intrinsics: np.ndarray,
    rotation_cw: np.ndarray | None = None,
    translation_cw: np.ndarray | None = None,
    min_score: float = 0.35,
    extra_lines: list[str] | None = None,
) -> np.ndarray:
    canvas = frame.copy()

    if result.keypoints2d is not None:
        if result.keypoints2d.bbox is not None:
            draw_bbox(canvas, result.keypoints2d.bbox)
        draw_keypoints2d(canvas, result.keypoints2d.xy, result.keypoints2d.scores, min_score)

    if result.skeleton3d is not None and rotation_cw is not None and translation_cw is not None:
        draw_reprojection(canvas, result.skeleton3d.xyz, intrinsics, rotation_cw, translation_cw)

    lines = list(extra_lines or [])
    if result.timings:
        lines.append(
            " ".join(
                f"{name.removesuffix('_ms')} {value:.1f}ms"
                for name, value in result.timings.items()
            )
        )
    if result.targets:
        stale = sum(1 for t in result.targets if t.stale)
        lines.append(f"trackers {len(result.targets)}" + (f" ({stale} stale)" if stale else ""))
    draw_stats(canvas, lines)

    return canvas


def lowest_confidence_joints(scores: np.ndarray, count: int = 3) -> list[str]:
    """Names of the least confident joints, for diagnosing tracking failures."""
    order = np.argsort(scores)[:count]
    return [f"{KEYPOINT_NAMES[i]}={scores[i]:.2f}" for i in order]
