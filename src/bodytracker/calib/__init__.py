"""Calibration: where the camera is, and how the user is proportioned."""

from .anchors import ANCHOR_BY_NAME, ANCHORS, Anchor

__all__ = [
    "ANCHORS",
    "ANCHOR_BY_NAME",
    "Anchor",
]

_LAZY = {
    "BodyCalibration": ("body", "BodyCalibration"),
    "calibrate_body": ("body", "calibrate_body"),
    "load_body": ("body", "load_body"),
    "save_body": ("body", "save_body"),
    "AutoCalibrator": ("auto", "AutoCalibrator"),
    "CameraCalibration": ("auto", "CameraCalibration"),
    "load_calibration": ("auto", "load_calibration"),
    "save_calibration": ("auto", "save_calibration"),
    "solve_camera": ("auto", "solve_camera"),
}


def __getattr__(name: str):
    # Deferred so importing this package does not require opencv or scipy.
    if name in _LAZY:
        module_name, attribute = _LAZY[name]
        import importlib

        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
