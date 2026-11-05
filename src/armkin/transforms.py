"""Homogeneous transforms and SO(3) helpers."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def dh_transform(a: float, alpha: float, d: float, theta: float) -> Array:
    """Standard (distal) Denavit-Hartenberg transform: Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rot_x(angle: float) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(angle: float) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(angle: float) -> Array:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def make_pose(R: Array, p: Array) -> Array:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def rotation_log(R: Array) -> Array:
    """Return the rotation vector (axis * angle) of rotation matrix ``R``.

    Handles the small-angle and near-pi cases explicitly.
    """
    cos_angle = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-9:
        # First-order approximation: R ~ I + [w]x
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - angle < 1e-6:
        # R ~ 2 n n^T - I ; pick the most numerically stable column.
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        axis = M[:, i] / np.sqrt(M[i, i])
        return axis * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * angle / (2.0 * np.sin(angle))


def pose_error(T_current: Array, T_target: Array) -> Array:
    """6-vector [position error; orientation error] expressed in the base frame."""
    dp = T_target[:3, 3] - T_current[:3, 3]
    dR = T_target[:3, :3] @ T_current[:3, :3].T
    return np.concatenate([dp, rotation_log(dR)])
