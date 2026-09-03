"""Reading BEDLAM 2.0 into training pairs.

BEDLAM is a synthetic dataset of rendered people with SMPL-X ground truth, shot
at 1280x720, which is the resolution a webcam claims to deliver. That makes it
an unusually good match for this project, and the licence is non-commercial
research use, which is what this is.

The lifter trains on 2D keypoints plus headset poses, not on pixels. The
ground-truth download is a few gigabytes; the images are twelve terabytes and
are only needed later, to measure how the 2D front-end actually degrades.
"""

from .devices import from_skeleton
from .joints import detect_layout, from_openpose_body25, from_smplx_body, to_halpe
from .names import SceneName, parse, try_parse
from .noise import KeypointNoise, fit_sigma
from .samples import (
    PoseSequence,
    SampleWindow,
    from_scene,
    load_shard,
    resample,
    save_shard,
    windows,
)
from .subset import Criteria, Scored, read_scene_list, report, scan_download, score, select

__all__ = [
    "Criteria",
    "KeypointNoise",
    "PoseSequence",
    "SampleWindow",
    "SceneName",
    "Scored",
    "detect_layout",
    "fit_sigma",
    "from_openpose_body25",
    "from_scene",
    "from_skeleton",
    "from_smplx_body",
    "load_shard",
    "parse",
    "read_scene_list",
    "report",
    "resample",
    "save_shard",
    "scan_download",
    "score",
    "select",
    "to_halpe",
    "try_parse",
    "windows",
]
