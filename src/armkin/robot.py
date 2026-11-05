"""Serial manipulator model: forward kinematics, Jacobian, inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .transforms import Array, dh_transform, pose_error


@dataclass(frozen=True)
class Link:
    """One link in standard DH convention.

    For a revolute joint the joint variable is added to ``theta``; for a prismatic
    joint it is added to ``d``.
    """

    a: float
    alpha: float
    d: float = 0.0
    theta: float = 0.0
    prismatic: bool = False
    limits: Tuple[float, float] = (-np.pi, np.pi)

    def transform(self, q: float) -> Array:
        if self.prismatic:
            return dh_transform(self.a, self.alpha, self.d + q, self.theta)
        return dh_transform(self.a, self.alpha, self.d, self.theta + q)


@dataclass(frozen=True)
class IKResult:
    q: Array
    success: bool
    iterations: int
    position_error: float
    orientation_error: float
    restarts: int = 0


@dataclass
class SerialRobot:
    links: List[Link]
    name: str = "robot"
    base: Array = field(default_factory=lambda: np.eye(4))
    tool: Array = field(default_factory=lambda: np.eye(4))

    @property
    def dof(self) -> int:
        return len(self.links)

    @property
    def lower(self) -> Array:
        return np.array([lk.limits[0] for lk in self.links])

    @property
    def upper(self) -> Array:
        return np.array([lk.limits[1] for lk in self.links])

    def _check(self, q: ArrayLike) -> Array:
        arr = np.asarray(q, dtype=float)
        if arr.shape != (self.dof,):
            raise ValueError(f"expected {self.dof} joint values, got shape {arr.shape}")
        return arr

    def within_limits(self, q: ArrayLike, tol: float = 1e-9) -> bool:
        arr = self._check(q)
        return bool(np.all(arr >= self.lower - tol) and np.all(arr <= self.upper + tol))

    # ------------------------------------------------------------------ FK
    def fk_all(self, q: ArrayLike) -> List[Array]:
        """Frames 0..n (base to flange); the tool transform is applied to the last frame."""
        frames = [self.base.copy()]
        T = self.base.copy()
        for link, qi in zip(self.links, self._check(q)):
            T = T @ link.transform(qi)
            frames.append(T)
        frames[-1] = frames[-1] @ self.tool
        return frames

    def fk(self, q: ArrayLike) -> Array:
        return self.fk_all(q)[-1]

    # ------------------------------------------------------------ Jacobian
    def jacobian(self, q: ArrayLike) -> Array:
        """Geometric Jacobian (6 x n) in the base frame: rows = [v; w]."""
        frames = self.fk_all(q)
        p_end = frames[-1][:3, 3]
        J = np.zeros((6, self.dof))
        for i, link in enumerate(self.links):
            z = frames[i][:3, 2]
            o = frames[i][:3, 3]
            if link.prismatic:
                J[:3, i] = z
            else:
                J[:3, i] = np.cross(z, p_end - o)
                J[3:, i] = z
        return J

    def manipulability(self, q: ArrayLike, position_only: bool = False) -> float:
        """Yoshikawa manipulability: product of the Jacobian's singular values.

        Equals sqrt(det(J J^T)) for redundant arms and sqrt(det(J^T J)) for arms with
        fewer joints than task dimensions (e.g. a planar arm); ~0 near singularities.
        """
        J = self.jacobian(q)
        if position_only:
            J = J[:3]
        return float(np.prod(np.linalg.svd(J, compute_uv=False)))

    # ------------------------------------------------------------------ IK
    def ik(
        self,
        target: ArrayLike,
        q0: Optional[ArrayLike] = None,
        *,
        position_only: bool = False,
        tol_pos: float = 1e-4,
        tol_rot: float = 1e-3,
        max_iter: int = 200,
        damping: float = 0.05,
        max_step: float = 0.3,
        restarts: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> IKResult:
        """Damped least-squares (Levenberg-Marquardt style) numerical IK.

        ``target`` is a 4x4 pose, or a 3-vector position when ``position_only``.
        Joint limits are enforced by clipping after every step. With ``restarts > 0``
        the solver retries from random joint configurations inside the limits.
        """
        rng = rng or np.random.default_rng(0)
        target_arr = np.asarray(target, dtype=float)
        if position_only:
            T_target = np.eye(4)
            T_target[:3, 3] = target_arr[:3, 3] if target_arr.shape == (4, 4) else target_arr
        else:
            T_target = target_arr
            if T_target.shape != (4, 4):
                raise ValueError("full-pose IK needs a 4x4 target")

        starts = [self._check(q0) if q0 is not None else np.zeros(self.dof)]
        starts += [rng.uniform(self.lower, self.upper) for _ in range(restarts)]

        best: Optional[IKResult] = None
        for attempt, q_start in enumerate(starts):
            result = self._ik_once(
                T_target, q_start, position_only, tol_pos, tol_rot, max_iter, damping, max_step
            )
            result = IKResult(
                result.q, result.success, result.iterations,
                result.position_error, result.orientation_error, attempt,
            )
            if result.success:
                return result
            if best is None or result.position_error < best.position_error:
                best = result
        assert best is not None
        return best

    def _ik_once(
        self,
        T_target: Array,
        q: Array,
        position_only: bool,
        tol_pos: float,
        tol_rot: float,
        max_iter: int,
        damping: float,
        max_step: float,
    ) -> IKResult:
        q = np.clip(q.astype(float).copy(), self.lower, self.upper)
        rows = slice(0, 3) if position_only else slice(0, 6)
        lam2 = damping**2
        for it in range(1, max_iter + 1):
            e = pose_error(self.fk(q), T_target)
            ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if ep < tol_pos and (position_only or er < tol_rot):
                return IKResult(q, True, it - 1, ep, 0.0 if position_only else er)
            J = self.jacobian(q)[rows]
            err = e[rows]
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(J.shape[0]), err)
            norm = np.linalg.norm(dq)
            if norm > max_step:
                dq *= max_step / norm
            q = np.clip(q + dq, self.lower, self.upper)
        e = pose_error(self.fk(q), T_target)
        ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
        ok = ep < tol_pos and (position_only or er < tol_rot)
        return IKResult(q, ok, max_iter, ep, 0.0 if position_only else er)
