"""Lifting 2D keypoints to metric 3D."""

from .anchored import AnchoredLifter, anchor_points_play
from .base import LiftContext, Lifter
from .geometric import GeometricLifter
from .kinematics import (
    backproject_rays,
    bone_length_lookup,
    bone_lengths_from_height,
    build_adjacency,
    enforce_bone_lengths,
    estimate_root_depth,
    lift_by_bone_lengths,
    measure_bone_lengths,
    propagate_depths,
    solve_child_depth,
)

__all__ = [
    "AnchoredLifter",
    "GeometricLifter",
    "LiftContext",
    "Lifter",
    "NeuralLifter",
    "anchor_points_play",
    "backproject_rays",
    "bone_length_lookup",
    "bone_lengths_from_height",
    "build_adjacency",
    "enforce_bone_lengths",
    "estimate_root_depth",
    "lift_by_bone_lengths",
    "measure_bone_lengths",
    "propagate_depths",
    "solve_child_depth",
]


def __getattr__(name: str):
    # Deferred so importing this package does not require onnxruntime.
    if name == "NeuralLifter":
        from .neural import NeuralLifter

        return NeuralLifter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
