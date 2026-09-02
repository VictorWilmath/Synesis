"""The One Euro filter.

Casiez, Roussel and Vogel (CHI 2012). Chosen over a Kalman filter because the
tradeoff it exposes is exactly the one that matters for interactive tracking:
at low speeds it filters hard to kill jitter, and as speed rises it backs off
to avoid adding lag. A fixed-gain filter has to pick one or the other, and
either choice is visible in VR.

Two knobs:

``min_cutoff``
    Lower means smoother when still, at the cost of lag. Start here.
``beta``
    Higher means the filter opens up faster as you move, reducing lag on fast
    motion at the cost of letting some jitter through. Raise it if tracking
    feels like it is dragging behind you.
"""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    """Smoothing factor for a first-order low pass at a given cutoff."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class _LowPass:
    """First-order low pass over an array, with per-element state."""

    def __init__(self) -> None:
        self.value: np.ndarray | None = None

    def __call__(self, x: np.ndarray, alpha: float) -> np.ndarray:
        if self.value is None:
            self.value = np.array(x, dtype=np.float64, copy=True)
        else:
            self.value = alpha * x + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class OneEuroFilter:
    """Adaptive low-pass over a fixed-shape array signal.

    Operates element-wise, so a single instance can filter an entire (26, 3)
    skeleton with independent adaptation per coordinate.
    """

    def __init__(
        self,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.35,
        d_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoffs must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._prev: np.ndarray | None = None
        self._last_time: float | None = None

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._prev = None
        self._last_time = None

    def __call__(self, value: np.ndarray, timestamp: float) -> np.ndarray:
        x = np.asarray(value, dtype=np.float64)

        if self._last_time is None:
            self._last_time = timestamp
            self._prev = x.copy()
            self._x(x, 1.0)
            return x.copy()

        dt = timestamp - self._last_time
        if dt <= 0:
            # Duplicate or out-of-order timestamp; hold the current estimate
            # rather than dividing by zero.
            return self._x.value.copy() if self._x.value is not None else x.copy()
        self._last_time = timestamp

        derivative = (x - self._prev) / dt
        self._prev = x.copy()
        edx = self._dx(derivative, _alpha(self.d_cutoff, dt))

        # Speed opens the cutoff up, trading smoothing for responsiveness.
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        alpha = 1.0 / (1.0 + tau / dt)

        if self._x.value is None:
            self._x.value = x.copy()
        else:
            self._x.value = alpha * x + (1.0 - alpha) * self._x.value
        return self._x.value.copy()
