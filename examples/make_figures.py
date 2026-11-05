"""Generate the README figures in assets/ (requires matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from armkin import planar_2r, planar_3r, quintic  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets"
INK, ACCENT, MUTED = "#0b0d12", "#2554d8", "#9aa3b2"


def reach_figure() -> None:
    arm = planar_3r()
    target = np.array([0.6, 1.4, 0.0])
    q0 = np.array([0.0, 0.3, 0.3])
    res = arm.ik(target, q0=q0, position_only=True)
    traj = quintic(q0, res.q, duration=2.0, n=60)

    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    for k, q in enumerate(traj.q[::6]):
        pts = np.array([T[:2, 3] for T in arm.fk_all(q)])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=MUTED, lw=1, alpha=0.25 + 0.6 * k / 10)
    path = np.array([arm.fk(q)[:2, 3] for q in traj.q])
    ax.plot(path[:, 0], path[:, 1], "--", color=ACCENT, lw=1.2, label="end-effector path")
    for q, style in ((q0, MUTED), (res.q, INK)):
        pts = np.array([T[:2, 3] for T in arm.fk_all(q)])
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=style, lw=3, ms=5)
    ax.plot(*target[:2], "x", color=ACCENT, ms=10, mew=2, label="target")
    ax.set_title(f"Planar 3R: DLS IK ({res.iterations} iterations) + quintic trajectory", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "planar3r_reach.png")
    plt.close(fig)


def manipulability_figure() -> None:
    arm = planar_2r()
    q1 = np.linspace(-np.pi, np.pi, 181)
    q2 = np.linspace(-np.pi, np.pi, 181)
    W = np.array([[arm.manipulability([a, b], position_only=True) for a in q1] for b in q2])
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=150)
    im = ax.imshow(W, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi], cmap="Blues", aspect="auto")
    ax.set_xlabel("q1 [rad]")
    ax.set_ylabel("q2 [rad]")
    ax.set_title("Planar 2R manipulability (0 at q2 = 0, ±π: singular)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(ASSETS / "planar2r_manipulability.png")
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    reach_figure()
    manipulability_figure()
    print("wrote", *sorted(p.name for p in ASSETS.glob("*.png")))
