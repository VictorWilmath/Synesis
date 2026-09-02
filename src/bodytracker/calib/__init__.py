"""Calibration: where the camera is, and how the user is proportioned."""

from .anchors import (
    ANCHORS,
    Anchor,
    Correspondence,
    CorrespondenceBuffer,
    build_point_arrays,
    pack_offsets,
    unpack_offsets,
)

__all__ = [
    "ANCHORS",
    "Anchor",
    "Correspondence",
    "CorrespondenceBuffer",
    "build_point_arrays",
    "pack_offsets",
    "unpack_offsets",
]

_LAZY = {
    "BodyCalibration": ("body", "BodyCalibration"),
    "calibrate_body": ("body", "calibrate_body"),
    "load_body": ("body", "load_body"),
    "save_body": ("body", "save_body"),
    "CameraExtrinsics": ("extrinsics", "CameraExtrinsics"),
    "load_extrinsics": ("extrinsics", "load_extrinsics"),
    "save_extrinsics": ("extrinsics", "save_extrinsics"),
    "solve_extrinsics": ("extrinsics", "solve_extrinsics"),
    "OnlineExtrinsicsRefiner": ("online", "OnlineExtrinsicsRefiner"),
}


def __getattr__(name: str):
    # Deferred so importing this package does not require opencv or scipy.
    if name in _LAZY:
        module_name, attribute = _LAZY[name]
        import importlib

        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
