"""Joint-space trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .transforms import Array


@dataclass(frozen=True)
class Trajectory:
    t: Array  # (N,)
    q: Array  # (N, dof)
    qd: Array
    qdd: Array


def quintic(q_start: ArrayLike, q_end: ArrayLike, duration: float, n: int = 100) -> Trajectory:
    """Quintic polynomial with zero velocity and acceleration at both ends.

    s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5, tau = t / T.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if n < 2:
        raise ValueError("need at least two samples")
    q0 = np.asarray(q_start, dtype=float)
    q1 = np.asarray(q_end, dtype=float)
    if q0.shape != q1.shape:
        raise ValueError("start and end must have the same shape")
    t = np.linspace(0.0, duration, n)
    tau = t / duration
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / duration**2
    dq = q1 - q0
    return Trajectory(t, q0 + np.outer(s, dq), np.outer(sd, dq), np.outer(sdd, dq))
