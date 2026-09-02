"""Reading a BEDLAM scene folder name.

BEDLAM encodes what a scene contains in its directory name, which is the only
way to choose a subset without downloading terabytes first. The format is

    <date>_<bodies>_<sequences>_<body_batch>[_<token>...][_<fps>]

for example ``20221010_3-10_500_batch01hand_zoom_suburb_d_6fps``: 500 sequences
rendered on 2022-10-10, three to ten bodies in each, from the ``batch01hand``
body set, with a zooming camera in the ``suburb_d`` environment at 6fps.

The parsing is deliberately forgiving. These names were produced by a render
farm over a couple of years and they are not perfectly regular: at least one
scene has the body count welded onto the date with hyphens
(``20221020-3-8_250_...``). A parser that insisted on the documented form would
silently drop those scenes, and silently dropping training data is the kind of
bug that costs a week.

Sources: the scene lists in ``pixelite1201/BEDLAM`` (``bedlam_scene_names.csv``,
``bedlam2_scene_names.csv``) and the render configs in
``PerceivingSystems/bedlam_render``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# Camera behaviour tokens. Only the first group holds the camera still, which
# is what a webcam on a shelf does.
STATIC_TOKENS: Final[frozenset[str]] = frozenset({"static", "staticloc"})
MOVING_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "zoom",
        "orbit",
        "dolly",
        "dollyz",
        "tracking",
        "lookat",
        "approach",
        "pan",
        "vcam",
        "vcamego",
    }
)
# Pitched cameras are stationary but aimed steeply, written as pitchUp52 etc.
PITCH_PATTERN: Final[re.Pattern[str]] = re.compile(r"^pitch(up|down)(\d+)$", re.IGNORECASE)

# Framing tokens. These change what part of the body is in shot, and two of
# them also mean the PNG is stored rotated 90 degrees.
FRAMING_TOKENS: Final[frozenset[str]] = frozenset({"closeup", "upperbody", "portrait"})
ROTATED_TOKENS: Final[frozenset[str]] = frozenset({"closeup", "portrait"})

# Environments, split by whether they are rooms. Taken from the comments in
# bedlam_render's sequence generation config, which is the only place the
# suburb_* scenes are disambiguated: a, b and c are a living room, a kitchen
# and a bedroom, while d is the street outside.
INDOOR_SCENES: Final[frozenset[str]] = frozenset(
    {
        "suburb_a",
        "suburb_b",
        "suburb_c",
        "bigoffice",
        "archvizui3",
        "highschoolgym",
        "yogastudio",
        "archmodelsvol8",
        "busstation",
    }
)
OUTDOOR_SCENES: Final[frozenset[str]] = frozenset(
    {
        "suburb_d",
        "stadium",
        "citysample",
        "citysamplenight",
        "rome",
        "chemicalplant",
        "yakohama",
        "middleeast",
        "hdri",
    }
)

_BODIES = re.compile(r"^(\d+)(?:-(\d+))?$")
_FPS = re.compile(r"^(\d+)fps$")
_DATE = re.compile(r"^(\d{8})")


@dataclass(frozen=True, slots=True)
class SceneName:
    """What a scene folder name says about its contents."""

    raw: str
    date: str
    bodies_min: int
    bodies_max: int
    sequences: int
    body_batch: str
    tokens: tuple[str, ...]
    fps: float | None

    @property
    def single_subject(self) -> bool:
        """Exactly one body, every sequence.

        A range means the count is drawn per sequence, so even ``1-3`` cannot
        be trusted to give a clean single-person scene.
        """
        return self.bodies_min == 1 and self.bodies_max == 1

    @property
    def camera_moves(self) -> bool:
        return any(token in MOVING_TOKENS for token in self.tokens)

    @property
    def camera_static(self) -> bool:
        """A camera that does not move during the sequence.

        Includes scenes with no camera token at all: those are the HDRI
        backdrop renders, which are shot from a fixed viewpoint. A pitched
        camera counts as static too, since the pitch is a fixed aim rather
        than a motion, and a webcam looking down from a shelf is exactly that.
        """
        if self.camera_moves:
            return False
        return True

    @property
    def pitch_deg(self) -> float:
        """How far the camera is aimed off horizontal, positive for upward."""
        for token in self.tokens:
            match = PITCH_PATTERN.match(token)
            if match:
                degrees = float(match.group(2))
                return degrees if match.group(1).lower() == "up" else -degrees
        return 0.0

    @property
    def framing(self) -> str | None:
        for token in self.tokens:
            if token in FRAMING_TOKENS:
                return token
        return None

    @property
    def full_body_framing(self) -> bool:
        return self.framing is None

    @property
    def rotated(self) -> bool:
        """Whether the rendered PNG is stored rotated 90 degrees clockwise."""
        return any(token in ROTATED_TOKENS for token in self.tokens)

    @property
    def scene(self) -> str | None:
        """The environment token, if the name carries one."""
        joined = "_".join(self.tokens)
        for known in sorted(INDOOR_SCENES | OUTDOOR_SCENES, key=len, reverse=True):
            if known in joined:
                return known
        return None

    @property
    def indoor(self) -> bool | None:
        """True indoors, False outdoors, None if the name does not say."""
        scene = self.scene
        if scene is None:
            return None
        return scene in INDOOR_SCENES

    def __str__(self) -> str:
        return self.raw


def parse(folder: str) -> SceneName:
    """Read a scene folder name. Raises ValueError if it is not one."""
    name = folder.strip().strip("/\\")
    parts = name.split("_")
    if not parts or not _DATE.match(parts[0]):
        raise ValueError(f"not a BEDLAM scene name: {folder!r}")

    # The body count normally sits in field 1, but at least one scene has it
    # hyphenated onto the date instead. Recover it from there rather than
    # dropping the scene.
    date_parts = parts[0].split("-")
    date = date_parts[0]
    rest = parts[1:]
    if len(date_parts) > 1 and _BODIES.match("-".join(date_parts[1:])):
        rest = ["-".join(date_parts[1:]), *rest]

    if not rest or not (bodies := _BODIES.match(rest[0])):
        raise ValueError(f"no body count in {folder!r}")
    bodies_min = int(bodies.group(1))
    bodies_max = int(bodies.group(2) or bodies.group(1))

    if len(rest) < 2 or not rest[1].isdigit():
        raise ValueError(f"no sequence count in {folder!r}")
    sequences = int(rest[1])
    body_batch = rest[2] if len(rest) > 2 else ""

    tokens = [t.lower() for t in rest[3:]]
    fps: float | None = None
    if tokens and (match := _FPS.match(tokens[-1])):
        fps = float(match.group(1))
        tokens = tokens[:-1]

    return SceneName(
        raw=name,
        date=date,
        bodies_min=bodies_min,
        bodies_max=bodies_max,
        sequences=sequences,
        body_batch=body_batch,
        tokens=tuple(tokens),
        fps=fps,
    )


def try_parse(folder: str) -> SceneName | None:
    try:
        return parse(folder)
    except ValueError:
        return None
