"""Animated GIF for the README: LQR only vs. Geometric + LQR on the figure-eight.

    python make_animation.py      ->  ../docs/images/demo.gif
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.transform import Rotation

from controllers import GeometricLQRController, LQROnlyController
from scenarios import figure_eight, no_wind
from simulate import run

T, DT, FPS, SPEED = 13.0, 0.002, 15, 1.5          # 13 s of flight shown at 1.5x speed
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "images", "demo.gif")
ARM = 0.18                                          # drawn arm length [m]

runs = [("LQR only", LQROnlyController(), "tab:red"),
        ("Geometric + LQR", GeometricLQRController(wn=(2.2, 2.9, 3.5)), "tab:blue")]
logs = [run(c, figure_eight, no_wind, T=T, dt=DT) for _, c, _ in runs]

# The drawing also needs the attitude R, so re-run the closed loop and keep R.
from quadrotor import Quadrotor
def run_with_R(ctrl):
    q = Quadrotor(m=ctrl.m, J=np.diag(ctrl.J)); P, R = [], []
    for i in range(int(T / DT)):
        p_d, v_d, yaw_d, a_d = figure_eight(i * DT)
        u1, tau, _ = ctrl(q.p, q.v, q.R, q.w, p_d, v_d, yaw_d, a_d)
        P.append(q.p.copy()); R.append(q.R.copy()); q.step(u1, tau, DT)
    return np.array(P), np.array(R)
traj = [run_with_R(c) for _, c, _ in runs]

n_frames = int(T * FPS / SPEED)
idx = np.linspace(0, len(traj[0][0]) - 1, n_frames).astype(int)
p_des = logs[0]["p_d"]

fig = plt.figure(figsize=(9, 4.4), dpi=80)
axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]
artists = []
for ax, (name, _, col) in zip(axes, runs):
    ax.plot(p_des[:, 0], p_des[:, 1], p_des[:, 2], "--", color="0.35", lw=1, label="Desired")
    path, = ax.plot([], [], [], color=col, lw=1.6, label="Actual")
    arm1, = ax.plot([], [], [], color="k", lw=2.2)
    arm2, = ax.plot([], [], [], color="k", lw=2.2)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.8, 0.8); ax.set_zlim(0, 1.4)
    ax.set_box_aspect((2.4, 1.6, 1.4)); ax.view_init(elev=28, azim=-60)
    ax.set_xlabel("x [m]", labelpad=-6); ax.set_ylabel("y [m]", labelpad=-6); ax.set_zlabel("z [m]", labelpad=-6)
    ax.tick_params(labelsize=7, pad=-2)
    ax.set_title(name, color=col, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    artists.append((path, arm1, arm2))
txt = fig.text(0.5, 0.03, "", ha="center", fontsize=10)
fig.suptitle("Figure-eight tracking from takeoff (Python reimplementation)", fontsize=11)
fig.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.08, wspace=0.0)


def update(f):
    k = idx[f]
    for (P, R), (path, arm1, arm2) in zip(traj, artists):
        path.set_data_3d(P[:k + 1, 0], P[:k + 1, 1], P[:k + 1, 2])
        c = P[k]
        for arm, d in ((arm1, np.array([1, 1, 0])), (arm2, np.array([1, -1, 0]))):
            v = R[k] @ (d / np.linalg.norm(d)) * ARM
            seg = np.array([c - v, c + v])
            arm.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
    txt.set_text(f"t = {k * DT:4.1f} s")
    return []


anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS)
anim.save(OUT, writer=PillowWriter(fps=FPS))
print(f"Saved {os.path.normpath(OUT)}")
