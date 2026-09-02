"""Reading BEDLAM 2.0 into training pairs.

BEDLAM is a synthetic dataset of rendered people with SMPL-X ground truth, shot
at 1280x720, which is the resolution a webcam claims to deliver. That makes it
an unusually good match for this project, and the licence is non-commercial
research use, which is what this is.
"""

from .names import SceneName, parse, try_parse
from .subset import Criteria, Scored, read_scene_list, report, scan_download, score, select

__all__ = [
    "Criteria",
    "SceneName",
    "Scored",
    "parse",
    "read_scene_list",
    "report",
    "scan_download",
    "score",
    "select",
    "try_parse",
]
