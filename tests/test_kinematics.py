from __future__ import annotations

import numpy as np
import pytest

from armkin import (
    make_pose,
    planar_2r,
    planar_2r_ik,
    planar_3r,
    pose_error,
    quintic,
    rot_x,
    rot_z,
    rotation_log,
    ur5,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------- transforms


@pytest.mark.parametrize("angle", [0.0, 1e-10, 0.3, 1.5, np.pi - 1e-8, np.pi])
def test_rotation_log_roundtrip(angle):
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    # Build R from Rodrigues' formula and recover axis * angle.
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    w = rotation_log(R)
    assert np.isclose(np.linalg.norm(w), angle, atol=1e-6)
    if angle > 1e-6:
        assert np.allclose(np.abs(w / np.linalg.norm(w)), np.abs(axis), atol=1e-5)


def test_pose_error_zero_for_same_pose():
    T = make_pose(rot_x(0.4) @ rot_z(-1.1), np.array([0.1, 0.2, 0.3]))
    assert np.allclose(pose_error(T, T), 0.0)


# ------------------------------------------------------------------------ FK


def test_planar_2r_fk_known_values():
    arm = planar_2r(1.0, 0.8)
    assert np.allclose(arm.fk([0, 0])[:3, 3], [1.8, 0, 0])
    assert np.allclose(arm.fk([np.pi / 2, 0])[:3, 3], [0, 1.8, 0])
    assert np.allclose(arm.fk([0, np.pi / 2])[:3, 3], [1.0, 0.8, 0])


def test_ur5_home_pose():
    # With all joints at zero the UR5 (standard DH) flange is at
    # x = a2 + a3, y = -(d4 + d6), z = d1 - d5.
    p = ur5().fk(np.zeros(6))[:3, 3]
    assert np.allclose(p, [-0.81725, -0.19145, -0.005491], atol=1e-6)


def test_wrong_joint_count_raises():
    with pytest.raises(ValueError):
        planar_2r().fk([0.0])


# ------------------------------------------------------------------ Jacobian


@pytest.mark.parametrize("make", [planar_3r, ur5])
def test_jacobian_matches_finite_differences(make):
    arm = make()
    q = RNG.uniform(-1.5, 1.5, arm.dof)
    J = arm.jacobian(q)
    eps = 1e-6
    T0 = arm.fk(q)
    for i in range(arm.dof):
        dq = np.zeros(arm.dof)
        dq[i] = eps
        T1 = arm.fk(q + dq)
        num = pose_error(T0, T1) / eps
        assert np.allclose(J[:, i], num, atol=1e-4)


def test_manipulability_zero_at_singularity():
    arm = planar_2r()
    assert arm.manipulability([0.3, 0.0], position_only=True) < 1e-9  # arm fully stretched
    assert arm.manipulability([0.3, 1.2], position_only=True) > 0.1


# ------------------------------------------------------------------------ IK


def test_planar_2r_analytic_ik_both_elbows():
    sols = planar_2r_ik(1.2, 0.6)
    assert len(sols) == 2
    arm = planar_2r()
    for q in sols:
        assert np.allclose(arm.fk(q)[:2, 3], [1.2, 0.6], atol=1e-9)


def test_planar_2r_analytic_ik_unreachable():
    assert planar_2r_ik(3.0, 0.0) == []


def test_numeric_ik_agrees_with_analytic_2r():
    arm = planar_2r()
    target = np.array([0.9, 0.9, 0.0])
    res = arm.ik(target, q0=[0.2, 0.5], position_only=True)
    assert res.success
    assert np.allclose(arm.fk(res.q)[:3, 3], target, atol=1e-4)
    analytic = planar_2r_ik(0.9, 0.9)
    assert any(np.allclose(res.q, s, atol=1e-3) for s in analytic)


def test_ur5_full_pose_ik_on_random_reachable_targets():
    arm = ur5()
    rng = np.random.default_rng(7)
    solved = 0
    for _ in range(20):
        q_true = rng.uniform(-np.pi, np.pi, 6)
        target = arm.fk(q_true)
        res = arm.ik(target, q0=np.zeros(6), restarts=5, rng=rng)
        if res.success:
            solved += 1
            assert np.linalg.norm(pose_error(arm.fk(res.q), target)[:3]) < 1e-4
            assert arm.within_limits(res.q)
    assert solved >= 18  # numerical IK is not guaranteed; require a high success rate


def test_ik_reports_failure_for_unreachable_target():
    res = planar_2r().ik(np.array([5.0, 0.0, 0.0]), position_only=True, max_iter=50)
    assert not res.success
    assert res.position_error > 3.0


def test_ik_respects_joint_limits():
    arm = planar_2r()
    from dataclasses import replace

    arm.links = [replace(lk, limits=(0.0, 1.0)) for lk in arm.links]
    res = arm.ik(np.array([-1.0, 0.5, 0.0]), position_only=True, max_iter=100)
    assert arm.within_limits(res.q)


# ---------------------------------------------------------------- trajectory


def test_quintic_boundary_conditions():
    traj = quintic([0.0, 1.0], [1.0, -1.0], duration=2.0, n=201)
    assert np.allclose(traj.q[0], [0, 1]) and np.allclose(traj.q[-1], [1, -1])
    assert np.allclose(traj.qd[[0, -1]], 0) and np.allclose(traj.qdd[[0, -1]], 0)
    # Peak velocity of the quintic profile is 1.875 * dq / T at the midpoint.
    assert np.allclose(np.abs(traj.qd[100]), 1.875 * np.array([1, 2]) / 2.0)


def test_quintic_rejects_bad_duration():
    with pytest.raises(ValueError):
        quintic([0], [1], duration=0)
