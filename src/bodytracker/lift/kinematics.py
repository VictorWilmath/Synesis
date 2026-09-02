"""Bone lengths and depth propagation along the kinematic tree.

The geometric core of the Phase 1 baseline, kept pure so it can be tested
without a camera, a headset or onnxruntime.

The idea: a pinhole camera fixes each joint's 3D position to a ray, leaving one
unknown per joint, its depth. A bone of known length between two joints
constrains the pair, giving a quadratic in the child's depth once the parent's
is known. Walking the tree outward from the pelvis therefore recovers a full
3D skeleton from 2D keypoints plus bone lengths.

The catch, and the reason this is only a baseline: each quadratic has two
roots, corresponding to the child being nearer or further than the parent.
A single camera genuinely cannot distinguish them. This module resolves the
ambiguity with a "stay near the parent" heuristic, which is right most of the
time and confidently wrong when a limb points at the camera. Fixing that
properly is what the trained lifter and the HMD anchor are for.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..skeleton import (
    BONE_LENGTH_PRIOR_RATIO,
    BONES,
    HIP,
    KINEMATIC_PARENT,
    LATERAL_PAIRS,
    LEFT_ANKLE,
    LEFT_BIG_TOE,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_SMALL_TOE,
    NECK,
    NUM_KEYPOINTS,
    RIGHT_ANKLE,
    RIGHT_BIG_TOE,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_SMALL_TOE,
    ROOT,
    TOPOLOGICAL_ORDER,
    VERTICAL_BONES,
)

# Depth below this is behind or inside the camera and cannot be valid.
_MIN_DEPTH = 0.2
_MAX_DEPTH = 12.0

# How far from vertical a bone may lean and still count as upright, as the sine
# of the angle. About twelve degrees.
_UPRIGHT_SINE = 0.21

# How much more upright the winner must be, as a fraction of bone length,
# before gravity is worth acting on. Below this the two candidates are
# effectively the same answer and the choice does not matter.
_UPRIGHT_MARGIN = 0.02


def _is_vertical_bone(a: int, b: int) -> bool:
    return (a, b) in VERTICAL_BONES or (b, a) in VERTICAL_BONES


_LATERAL_PAIR_BY_PARENT: dict[int, tuple[int, int]] = {
    parent: (left, right) for parent, left, right in LATERAL_PAIRS
}


def bone_lengths_from_height(height_m: float) -> dict[tuple[int, int], float]:
    """Scale the anthropometric prior to a given standing height."""
    return {bone: ratio * height_m for bone, ratio in BONE_LENGTH_PRIOR_RATIO.items()}


def bone_length_lookup(lengths: dict[tuple[int, int], float]) -> np.ndarray:
    """Per-joint length of the bone connecting it to its parent.

    Returns a (26,) array; the root and any joint whose bone is unknown get 0.
    """
    out = np.zeros(NUM_KEYPOINTS, dtype=np.float64)
    for (a, b), length in lengths.items():
        if KINEMATIC_PARENT[b] == a:
            out[b] = length
        elif KINEMATIC_PARENT[a] == b:
            out[a] = length
    return out


def measure_bone_lengths(xyz: np.ndarray) -> dict[tuple[int, int], float]:
    """Measure every bone in a 3D skeleton. Used by body calibration."""
    return {(a, b): float(np.linalg.norm(xyz[a] - xyz[b])) for a, b in BONES}


def backproject_rays(xy: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Turn pixel coordinates into normalised camera rays with z = 1.

    A joint at depth ``Z`` sits at ``Z * ray``.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    rays = np.ones((xy.shape[0], 3), dtype=np.float64)
    rays[:, 0] = (xy[:, 0] - cx) / fx
    rays[:, 1] = (xy[:, 1] - cy) / fy
    return rays


def estimate_root_depth(
    rays: np.ndarray,
    lengths: dict[tuple[int, int], float],
    scores: np.ndarray,
    min_score: float,
) -> float | None:
    """Estimate the pelvis depth from apparent bone sizes.

    For a bone of true length ``L`` whose endpoints are separated by ``d`` in
    the normalised image plane, the depth is ``L / d`` *if* the bone lies
    perpendicular to the view axis. Foreshortening can only shrink ``d``, so
    every bone yields an over-estimate and the true depth is bounded above by
    the smallest of them.

    Taking a low percentile rather than the strict minimum keeps that bound
    from being set by a single noisy keypoint.
    """
    estimates: list[float] = []
    for (a, b), length in lengths.items():
        if length <= 0 or scores[a] < min_score or scores[b] < min_score:
            continue
        separation = float(np.linalg.norm(rays[a, :2] - rays[b, :2]))
        if separation < 1e-6:
            continue
        estimates.append(length / separation)

    if len(estimates) < 3:
        return None

    depth = float(np.percentile(estimates, 20))
    return float(np.clip(depth, _MIN_DEPTH, _MAX_DEPTH))


def solve_child_depth(
    parent_depth: float,
    parent_ray: np.ndarray,
    child_ray: np.ndarray,
    length: float,
    prior: float | None = None,
) -> float:
    """Depth of a child joint given its parent's, from the bone-length constraint.

    Solves ``|Zc*rc - Zp*rp| = L`` for ``Zc``, which has two roots: the child
    can be nearer or further than the parent and project to the same pixel. A
    single camera cannot tell them apart, so the choice is a prior.

    With `prior` (in practice the previous frame's depth for this joint), the
    nearer root to it is taken, which resolves the ambiguity correctly as long
    as the limb did not flip between frames.

    Without one, the root nearer the parent is taken. That is a reasonable
    default for well-conditioned bones, but note the roots are symmetric about
    roughly the parent's depth whenever the bone points along the optical axis,
    so for a foreshortened bone this degenerates into a coin flip. A toe
    pointing at the camera is the usual victim.
    """
    a = float(np.dot(child_ray, child_ray))
    b = -2.0 * parent_depth * float(np.dot(child_ray, parent_ray))
    c = parent_depth**2 * float(np.dot(parent_ray, parent_ray)) - length**2

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        # The joints are further apart in the image than a bone this long can
        # reach, which means the length prior or the keypoints are wrong. Fall
        # back to the depth that minimises the 3D gap.
        return float(np.clip(-b / (2 * a), _MIN_DEPTH, _MAX_DEPTH))

    root = np.sqrt(discriminant)
    candidates = [(-b + root) / (2 * a), (-b - root) / (2 * a)]
    candidates = [z for z in candidates if z > _MIN_DEPTH]
    if not candidates:
        return float(np.clip(parent_depth, _MIN_DEPTH, _MAX_DEPTH))

    reference = parent_depth if prior is None else prior
    best = min(candidates, key=lambda z: abs(z - reference))
    return float(np.clip(best, _MIN_DEPTH, _MAX_DEPTH))


def child_depth_candidates(
    parent_depth: float,
    parent_ray: np.ndarray,
    child_ray: np.ndarray,
    length: float,
) -> list[float]:
    """Both valid roots of the bone-length constraint, nearest first.

    Exposed so callers with a better prior than "stay near the parent" can pick
    between them; see `resolve_foot_depths`.
    """
    a = float(np.dot(child_ray, child_ray))
    b = -2.0 * parent_depth * float(np.dot(child_ray, parent_ray))
    c = parent_depth**2 * float(np.dot(parent_ray, parent_ray)) - length**2

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return [float(np.clip(-b / (2 * a), _MIN_DEPTH, _MAX_DEPTH))]

    root = np.sqrt(discriminant)
    candidates = [z for z in ((-b + root) / (2 * a), (-b - root) / (2 * a)) if z > _MIN_DEPTH]
    if not candidates:
        return [float(np.clip(parent_depth, _MIN_DEPTH, _MAX_DEPTH))]
    return sorted(candidates, key=lambda z: abs(z - parent_depth))


def body_forward(xyz: np.ndarray) -> np.ndarray | None:
    """The direction the torso faces, from the hip line and the spine.

    Works in camera space as well as play space: both are right-handed, so
    ``cross(up, right)`` gives the facing direction in either. That means the
    feet can be disambiguated before the extrinsics are known.
    """
    right = xyz[RIGHT_HIP] - xyz[LEFT_HIP]
    up = xyz[NECK] - xyz[HIP]
    forward = np.cross(up, right)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        return None
    return forward / norm


def resolve_foot_depths(
    xyz: np.ndarray,
    rays: np.ndarray,
    lengths: dict[tuple[int, int], float],
) -> np.ndarray:
    """Re-solve the foot extremities using the direction the body faces.

    The toe and heel bones are short and, for a camera at roughly eye level,
    point almost straight down the optical axis. Both depth roots then place
    the joint the same distance from the ankle, just mirrored front to back, so
    no depth-based tie-break can separate them. Left alone the toes flip
    between frames, and since foot orientation is derived from exactly these
    keypoints, that flip is visible as a foot spinning on the spot.

    The torso, by contrast, is wide and upright and its facing direction is one
    of the better-conditioned things a single camera measures. Feet point
    roughly where the body points, so that resolves it. Not true when someone
    walks backwards, which is the documented cost of this prior and one of the
    things the trained lifter should do better.
    """
    forward = body_forward(xyz)
    if forward is None:
        return xyz

    # The hip line has already been resolved as a pair by this point, so it is
    # a reliable left/right axis for the feet.
    across = xyz[RIGHT_HIP] - xyz[LEFT_HIP]
    across_norm = float(np.linalg.norm(across))
    across = across / across_norm if across_norm > 1e-6 else None

    out = np.array(xyz, dtype=np.float64, copy=True)

    def length_of(a: int, b: int) -> float | None:
        return lengths.get((a, b), lengths.get((b, a)))

    for ankle, heel, big_toe, small_toe in (
        (LEFT_ANKLE, LEFT_HEEL, LEFT_BIG_TOE, LEFT_SMALL_TOE),
        (RIGHT_ANKLE, RIGHT_HEEL, RIGHT_BIG_TOE, RIGHT_SMALL_TOE),
    ):
        ankle_depth = float(out[ankle, 2])

        # Toes lead, heel trails.
        for joint, sign in ((big_toe, 1.0), (heel, -1.0)):
            bone = length_of(ankle, joint)
            if bone is None:
                continue
            candidates = child_depth_candidates(ankle_depth, rays[ankle], rays[joint], bone)
            if len(candidates) < 2:
                continue
            out[joint] = max(
                (rays[joint] * z for z in candidates),
                key=lambda point: sign * float(np.dot(point - out[ankle], forward)),
            )

        # The little toe sits on the outside of the foot, so the body's own
        # left/right axis says which way it should lie.
        bone = length_of(big_toe, small_toe)
        if bone is not None:
            outward = -across if ankle == LEFT_ANKLE else across
            candidates = child_depth_candidates(
                float(out[big_toe, 2]), rays[big_toe], rays[small_toe], bone
            )
            if outward is not None and len(candidates) == 2:
                out[small_toe] = max(
                    (rays[small_toe] * z for z in candidates),
                    key=lambda point: float(np.dot(point - out[big_toe], outward)),
                )
            else:
                out[small_toe] = rays[small_toe] * candidates[0]

    return out


def build_adjacency(
    lengths: dict[tuple[int, int], float],
) -> dict[int, list[tuple[int, float]]]:
    """Undirected bone graph, so the skeleton can be walked from any joint."""
    adjacency: dict[int, list[tuple[int, float]]] = {i: [] for i in range(NUM_KEYPOINTS)}
    for (a, b), length in lengths.items():
        if length <= 0:
            continue
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))
    return adjacency


def _choose_upright_depth(
    parent_point: np.ndarray,
    child_ray: np.ndarray,
    candidates: list[float],
    up_camera: np.ndarray,
    length: float,
) -> float | None:
    """Pick the root that keeps a hanging bone closest to the gravity axis.

    Both roots put the child the same distance from the parent, so the choice
    is between a bone that hangs straight down and one tilted along the viewing
    ray. For a spine or a thigh the upright answer is almost always the right
    one, and it discriminates far more sharply than "keep the child near the
    parent's depth", which fails outright whenever the camera is tilted: a
    slightly downward-looking camera sees a vertical spine as sloping away, so
    the most fronto-parallel solution is the mirrored, wrong one.

    Returns None unless one candidate is close to vertical outright. Merely
    being the more vertical of the two is not enough: a subject bent forty
    degrees forward has two candidates that both lean, and picking the less
    tilted one would straighten them up and be confidently wrong. Gravity is
    only informative in the upright case, which is most of them.
    """
    horizontal = []
    for depth in candidates:
        offset = child_ray * depth - parent_point
        horizontal.append(float(np.linalg.norm(offset - np.dot(offset, up_camera) * up_camera)))

    best = int(np.argmin(horizontal))
    if horizontal[best] > _UPRIGHT_SINE * length:
        return None
    if horizontal[1 - best] - horizontal[best] < _UPRIGHT_MARGIN * length:
        return None
    return candidates[best]


def _resolve_lateral_pair(
    parent_point: np.ndarray,
    left_ray: np.ndarray,
    right_ray: np.ndarray,
    left_candidates: list[float],
    right_candidates: list[float],
    up_camera: np.ndarray | None,
    facing_camera: np.ndarray | None,
) -> tuple[float, float]:
    """Choose shoulder or hip depths as a pair rather than one at a time.

    Two constraints, in order of strength:

    The pair must straddle its parent. A left shoulder and a right shoulder
    both solved to the near root end up on the same side of the neck, which no
    body does. Requiring the two offsets to point opposite ways eliminates
    those mixed solutions outright.

    That still leaves two self-consistent answers, mirror images of each other
    through the image plane, which is the classic monocular yaw ambiguity: from
    a low-resolution silhouette, turned slightly left and turned slightly right
    look the same. `facing_camera`, taken from the headset, breaks the tie.
    Without it the two remain genuinely indistinguishable.
    """
    best: tuple[float, float] = (left_candidates[0], right_candidates[0])
    best_score = np.inf

    for left_depth in left_candidates:
        for right_depth in right_candidates:
            left_point = left_ray * left_depth
            right_point = right_ray * right_depth
            left_offset = left_point - parent_point
            right_offset = right_point - parent_point

            left_norm = np.linalg.norm(left_offset)
            right_norm = np.linalg.norm(right_offset)
            if left_norm < 1e-9 or right_norm < 1e-9:
                continue
            score = float(np.linalg.norm(left_offset / left_norm + right_offset / right_norm))

            if facing_camera is not None and up_camera is not None:
                across = right_point - left_point
                forward = np.cross(up_camera, across)
                forward_norm = np.linalg.norm(forward)
                if forward_norm > 1e-9:
                    agreement = float(np.dot(forward / forward_norm, facing_camera))
                    score += 2.0 * (1.0 - agreement)

            if score < best_score:
                best_score = score
                best = (left_depth, right_depth)

    return best


def propagate_depths(
    rays: np.ndarray,
    root_depth: float,
    lengths: dict[tuple[int, int], float],
    depth_prior: np.ndarray | None = None,
    root_joint: int = ROOT,
    up_camera: np.ndarray | None = None,
    facing_camera: np.ndarray | None = None,
) -> np.ndarray:
    """Walk the skeleton outward from a pinned joint, solving depths in turn.

    Breadth-first over the undirected bone graph rather than down the kinematic
    tree, so the walk can start anywhere. That matters because the HMD pins the
    *head*, not the pelvis, and re-rooting at the head propagates the known
    depth outward exactly instead of searching for a pelvis depth that happens
    to put the head in the right place.

    `depth_prior` is a (26,) array of expected depths, normally the previous
    frame's solution. Supplying it is what keeps foreshortened limbs from
    flipping between the two valid solutions frame to frame.

    `up_camera` is the world's up axis expressed in camera space. Supplying it
    lets bones that hang along gravity be disambiguated by which candidate is
    more upright, which is a much better prior than depth proximity for the
    spine and the limbs.
    """
    depths = np.full(NUM_KEYPOINTS, np.nan, dtype=np.float64)
    depths[root_joint] = root_depth

    adjacency = build_adjacency(lengths)
    prior_shift = 0.0
    if depth_prior is not None and np.isfinite(depth_prior[root_joint]):
        # Offset the prior by however much the root moved, so a subject walking
        # toward the camera does not fight their own history.
        prior_shift = root_depth - float(depth_prior[root_joint])

    if up_camera is not None:
        up_camera = np.asarray(up_camera, dtype=np.float64)
        up_camera = up_camera / max(float(np.linalg.norm(up_camera)), 1e-9)

    if facing_camera is not None:
        facing_camera = np.asarray(facing_camera, dtype=np.float64)
        facing_camera = facing_camera / max(float(np.linalg.norm(facing_camera)), 1e-9)

    queue = deque([root_joint])
    while queue:
        current = queue.popleft()

        pair = _LATERAL_PAIR_BY_PARENT.get(current)
        if pair is not None and all(np.isnan(depths[j]) for j in pair):
            left, right = pair
            left_length = lengths.get((current, left), lengths.get((left, current)))
            right_length = lengths.get((current, right), lengths.get((right, current)))
            if left_length and right_length:
                parent_point = rays[current] * float(depths[current])
                left_depth, right_depth = _resolve_lateral_pair(
                    parent_point,
                    rays[left],
                    rays[right],
                    child_depth_candidates(
                        float(depths[current]), rays[current], rays[left], left_length
                    ),
                    child_depth_candidates(
                        float(depths[current]), rays[current], rays[right], right_length
                    ),
                    up_camera,
                    facing_camera,
                )
                depths[left] = float(np.clip(left_depth, _MIN_DEPTH, _MAX_DEPTH))
                depths[right] = float(np.clip(right_depth, _MIN_DEPTH, _MAX_DEPTH))
                queue.extend((left, right))

        for neighbour, length in adjacency[current]:
            if not np.isnan(depths[neighbour]):
                continue

            parent_depth = float(depths[current])
            prior = None
            if depth_prior is not None and np.isfinite(depth_prior[neighbour]):
                prior = float(depth_prior[neighbour] + prior_shift)

            chosen = None
            # Gravity is a guess about the pose; the previous frame is a
            # measurement of it. Only guess when there is nothing to remember,
            # which in practice means the first frame after a reset. Getting
            # that frame right matters more than it sounds, because every
            # later frame inherits it through the temporal prior.
            if prior is None and up_camera is not None and _is_vertical_bone(current, neighbour):
                candidates = child_depth_candidates(
                    parent_depth, rays[current], rays[neighbour], length
                )
                if len(candidates) == 2:
                    chosen = _choose_upright_depth(
                        rays[current] * parent_depth,
                        rays[neighbour],
                        candidates,
                        up_camera,
                        length,
                    )

            if chosen is None:
                chosen = solve_child_depth(
                    parent_depth, rays[current], rays[neighbour], length, prior=prior
                )

            depths[neighbour] = float(np.clip(chosen, _MIN_DEPTH, _MAX_DEPTH))
            queue.append(neighbour)

    # Joints with no bones of their own, i.e. the face, inherit from up the
    # kinematic tree.
    for joint in TOPOLOGICAL_ORDER:
        if not np.isnan(depths[joint]):
            continue
        parent = KINEMATIC_PARENT[joint]
        depths[joint] = depths[parent] if parent is not None else root_depth

    if np.isnan(depths).any():
        depths = np.nan_to_num(depths, nan=root_depth)
    return depths


def lift_by_bone_lengths(
    xy: np.ndarray,
    scores: np.ndarray,
    intrinsics: np.ndarray,
    lengths: dict[tuple[int, int], float],
    min_score: float = 0.3,
    root_depth: float | None = None,
    depth_prior: np.ndarray | None = None,
    root_joint: int = ROOT,
    resolve_feet: bool = True,
    up_camera: np.ndarray | None = None,
    facing_camera: np.ndarray | None = None,
) -> np.ndarray | None:
    """Lift 2D keypoints to camera-space 3D using bone lengths alone.

    Pass `root_depth` to pin a joint at a known distance, which is what the HMD
    anchor does once extrinsics are known; otherwise it is estimated from
    apparent bone sizes. `root_joint` selects which joint is pinned, so the
    anchored lifter can pin the head instead of the pelvis.

    Pass `depth_prior`, normally the previous frame's depths, to stabilise the
    two-root ambiguity on foreshortened limbs.

    Returns an (N, 3) array in camera space, or None if the estimate failed.
    """
    rays = backproject_rays(np.asarray(xy, dtype=np.float64), intrinsics)

    if root_depth is None:
        root_depth = estimate_root_depth(rays, lengths, scores, min_score)
        if root_depth is None:
            return None

    depths = propagate_depths(
        rays,
        root_depth,
        lengths,
        depth_prior,
        root_joint=root_joint,
        up_camera=up_camera,
        facing_camera=facing_camera,
    )
    xyz = rays * depths[:, None]

    if resolve_feet:
        xyz = resolve_foot_depths(xyz, rays, lengths)
    return xyz


def enforce_bone_lengths(
    xyz: np.ndarray,
    lengths: dict[tuple[int, int], float],
    iterations: int = 4,
    strength: float = 1.0,
) -> np.ndarray:
    """Project a skeleton back onto its bone-length constraints.

    A Jacobi-style relaxation: every bone is corrected toward its target length
    at once and the corrections are averaged per joint, which converges more
    evenly than sweeping the tree in order. Prevents limbs from stretching when
    the lifter and the filters disagree.
    """
    points = np.array(xyz, dtype=np.float64, copy=True)
    bones = [(a, b, length) for (a, b), length in lengths.items() if length > 0]
    if not bones:
        return points

    # Vectorised over bones: this runs on every frame at capture rate, and the
    # per-bone Python loop it replaces cost more than the rest of the lift.
    ends_a = np.array([a for a, _, _ in bones])
    ends_b = np.array([b for _, b, _ in bones])
    targets = np.array([length for _, _, length in bones])

    counts = np.zeros(len(points))
    np.add.at(counts, ends_a, 1.0)
    np.add.at(counts, ends_b, 1.0)
    moved = counts > 0
    divisor = np.where(moved, counts, 1.0)[:, None]

    for _ in range(iterations):
        delta = points[ends_b] - points[ends_a]
        distance = np.linalg.norm(delta, axis=1)
        safe = distance > 1e-9

        shift = np.zeros_like(delta)
        scale = np.zeros_like(distance)
        scale[safe] = (distance[safe] - targets[safe]) / distance[safe] * 0.5 * strength
        shift = delta * scale[:, None]

        correction = np.zeros_like(points)
        np.add.at(correction, ends_a, shift)
        np.add.at(correction, ends_b, -shift)
        points += np.where(moved[:, None], correction / divisor, 0.0)

    return points
