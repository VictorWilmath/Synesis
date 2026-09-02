"""Bounding-box bookkeeping between detector runs.

With one static camera and one subject, running a person detector on every
frame is nearly pure overhead: it is the most expensive stage in the pipeline
and it re-answers a question that barely changed. Instead the box is carried
forward from the previous frame's keypoints, and the detector only runs
periodically or when tracking visibly fails.

Pure numpy, deliberately, so this is testable without onnxruntime present.
"""

from __future__ import annotations

import numpy as np

from ..skeleton import NUM_KEYPOINTS


def clip_bbox(bbox: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clamp an xyxy box to the frame."""
    x1, y1, x2, y2 = bbox
    return np.array(
        [
            np.clip(x1, 0, width - 1),
            np.clip(y1, 0, height - 1),
            np.clip(x2, 0, width - 1),
            np.clip(y2, 0, height - 1),
        ],
        dtype=np.float64,
    )


def bbox_area(bbox: np.ndarray) -> float:
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two xyxy boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(a) + bbox_area(b) - intersection
    return float(intersection / union) if union > 0 else 0.0


def bbox_from_keypoints(
    xy: np.ndarray,
    scores: np.ndarray,
    *,
    min_score: float,
    padding: float,
    frame_width: int,
    frame_height: int,
) -> np.ndarray | None:
    """Derive the next frame's crop from this frame's keypoints.

    Padding is proportional to the subject's size rather than a fixed pixel
    count, so the box stays valid whether the user is near the camera or across
    the room. Returns None when too few keypoints are confident enough to
    define a box.
    """
    if xy.shape != (NUM_KEYPOINTS, 2) or scores.shape != (NUM_KEYPOINTS,):
        raise ValueError("expected Halpe26-shaped keypoints")

    confident = scores >= min_score
    if confident.sum() < 4:
        return None

    points = xy[confident]
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)

    # Guard against a degenerate box when the subject is far away or the
    # confident keypoints happen to be collinear.
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    pad_x = width * padding
    pad_y = height * padding

    return clip_bbox(
        np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]),
        frame_width,
        frame_height,
    )


def select_person(
    bboxes: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    previous: np.ndarray | None = None,
) -> np.ndarray | None:
    """Choose which detection is the user.

    Continuity wins: if a detection overlaps the previous box, take it, so a
    person walking past in the background cannot steal the track. Otherwise
    fall back to a score combining size and centrality, since the user is
    normally the large, roughly centred subject.
    """
    if bboxes is None or len(bboxes) == 0:
        return None

    boxes = np.asarray(bboxes, dtype=np.float64)[:, :4]

    if previous is not None:
        overlaps = np.array([iou(box, previous) for box in boxes])
        best = int(np.argmax(overlaps))
        if overlaps[best] > 0.2:
            return boxes[best]

    frame_centre = np.array([frame_width / 2.0, frame_height / 2.0])
    diagonal = float(np.hypot(frame_width, frame_height))
    frame_area = float(frame_width * frame_height)

    scores = []
    for box in boxes:
        centre = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])
        centrality = 1.0 - min(1.0, float(np.linalg.norm(centre - frame_centre)) / (diagonal / 2))
        size = bbox_area(box) / frame_area
        scores.append(0.65 * size + 0.35 * centrality)

    return boxes[int(np.argmax(scores))]


def valid_ratio(scores: np.ndarray, min_score: float) -> float:
    """Fraction of keypoints confident enough to trust."""
    return float(np.mean(scores >= min_score))


def expand_to_aspect(bbox: np.ndarray, aspect: float, width: int, height: int) -> np.ndarray:
    """Grow a box to a target width/height ratio without cropping anything out.

    RTMPose warps its crop to a fixed 192x256. Feeding it a box of a different
    aspect ratio means the warp squashes the subject, which measurably costs
    keypoint accuracy, so the box is grown (never shrunk) to match.
    """
    x1, y1, x2, y2 = bbox
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    if box_w / box_h > aspect:
        box_h = box_w / aspect
    else:
        box_w = box_h * aspect

    return clip_bbox(
        np.array([cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2]),
        width,
        height,
    )
