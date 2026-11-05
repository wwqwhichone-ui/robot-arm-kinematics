"""Ready-made robot models.

The UR5 parameters are the standard DH values published by Universal Robots
(https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .robot import Link, SerialRobot

PI = np.pi


def planar_2r(l1: float = 1.0, l2: float = 0.8) -> SerialRobot:
    return SerialRobot([Link(a=l1, alpha=0.0), Link(a=l2, alpha=0.0)], name="planar-2R")


def planar_3r(l1: float = 1.0, l2: float = 0.8, l3: float = 0.4) -> SerialRobot:
    return SerialRobot(
        [Link(a=l1, alpha=0.0), Link(a=l2, alpha=0.0), Link(a=l3, alpha=0.0)], name="planar-3R"
    )


def ur5() -> SerialRobot:
    lim = (-2 * PI, 2 * PI)
    return SerialRobot(
        [
            Link(a=0.0, alpha=PI / 2, d=0.089159, limits=lim),
            Link(a=-0.425, alpha=0.0, d=0.0, limits=lim),
            Link(a=-0.39225, alpha=0.0, d=0.0, limits=lim),
            Link(a=0.0, alpha=PI / 2, d=0.10915, limits=lim),
            Link(a=0.0, alpha=-PI / 2, d=0.09465, limits=lim),
            Link(a=0.0, alpha=0.0, d=0.0823, limits=lim),
        ],
        name="UR5",
    )


def planar_2r_ik(x: float, y: float, l1: float = 1.0, l2: float = 0.8) -> List[Tuple[float, float]]:
    """Closed-form IK for a planar 2R arm. Returns 0, 1 or 2 solutions (elbow down / up)."""
    r2 = x * x + y * y
    c2 = (r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if c2 < -1 - 1e-12 or c2 > 1 + 1e-12:
        return []
    c2 = float(np.clip(c2, -1.0, 1.0))
    sols = []
    for s2 in sorted({np.sqrt(1 - c2 * c2), -np.sqrt(1 - c2 * c2)}, reverse=True):
        q2 = np.arctan2(s2, c2)
        q1 = np.arctan2(y, x) - np.arctan2(l2 * s2, l1 + l2 * c2)
        sols.append((float(np.arctan2(np.sin(q1), np.cos(q1))), float(q2)))
    return sols
