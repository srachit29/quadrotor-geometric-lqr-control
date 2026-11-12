"""Closed-loop simulation, metrics and plots."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quadrotor import Quadrotor


def run(controller, trajectory, wind, T=16.0, dt=0.002, p0=(0.0, 0.0, 0.07)):
    quad = Quadrotor(m=controller.m, J=np.diag(controller.J), p0=p0)
    n = int(T / dt)
    log = {k: [] for k in ("t", "p", "p_d", "v", "v_d", "e_R", "e_w", "u1", "tau", "wind")}
    for i in range(n):
        t = i * dt
        p_d, v_d, yaw_d, a_d = trajectory(t)
        u1, tau, info = controller(quad.p, quad.v, quad.R, quad.w, p_d, v_d, yaw_d, a_d)
        F_w = wind(t)
        for k, val in (("t", t), ("p", quad.p.copy()), ("p_d", p_d), ("v", quad.v.copy()), ("v_d", v_d),
                       ("e_R", info["e_R"]), ("e_w", info["e_w"]), ("u1", u1), ("tau", tau), ("wind", F_w)):
            log[k].append(val)
        quad.step(u1, tau, dt, F_w)
    return {k: np.array(v) for k, v in log.items()}


def rmse(e):
    return float(np.sqrt(np.mean(np.sum(e**2, axis=1))))


def metrics(log):
    e_p = log["p"] - log["p_d"]
    e_v = log["v"] - log["v_d"]
    return {
        "pos_rmse": rmse(e_p),
        "vel_rmse": rmse(e_v),
        "att_rmse": rmse(log["e_R"]),
        "max_att_err": float(np.max(np.abs(log["e_R"][:, :2]))),
        "max_pos_err": np.max(np.abs(e_p), axis=0),
        "max_thrust": float(np.max(log["u1"])),
        "mean_thrust": float(np.mean(log["u1"])),
        "max_torque": np.max(np.abs(log["tau"]), axis=0),
    }


def print_report(title, ctrl, m):
    print("\n" + "=" * 70)
    print(f"📊 SIMULATION ANALYSIS REPORT ({title})")
    print("=" * 70)
    print(f"Vehicle mass: {ctrl.m} kg, inertia: {np.diag(ctrl.J)}")
    if hasattr(ctrl, "kx"):
        print(f"Outer loop: wn = {ctrl.wn} rad/s, kx = {np.round(np.diag(ctrl.kx), 3)}, kv = {np.round(np.diag(ctrl.kv), 3)}")
    print(f"LQR Q = {np.diag(ctrl.Q)}, R = {np.diag(ctrl.R_lqr)}")
    print(f"Position RMSE: {m['pos_rmse']:.4f} m   (max |e| XYZ: {np.round(m['max_pos_err'], 3)} m)")
    print(f"Velocity RMSE: {m['vel_rmse']:.4f} m/s")
    print(f"Attitude RMSE: {m['att_rmse']:.4f} rad (max roll/pitch error {m['max_att_err']:.3f} rad)")
    print(f"Thrust: mean {m['mean_thrust']:.2f} N, max {m['max_thrust']:.2f} N (hover {ctrl.m * 9.81:.2f} N)")
    print(f"Max |torque| (x, y, z): {np.round(m['max_torque'], 3)} N m")


def plot_tracking(logs, title, path):
    """logs: dict label -> log. Same 2x2 layout as the Isaac Sim plots."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    first = next(iter(logs.values()))
    t, p_d = first["t"], first["p_d"]
    for ax, i, name in ((axs[0, 0], 0, "X"), (axs[0, 1], 1, "Y"), (axs[1, 0], 2, "Z")):
        ax.plot(t, p_d[:, i], "--", color="black", lw=1.2, label="Desired")
        for label, log in logs.items():
            ax.plot(log["t"], log["p"][:, i], label=label)
        ax.set_title(f"{name} position"); ax.set_xlabel("Time [s]"); ax.set_ylabel("[m]")
        ax.grid(True, ls=":"); ax.legend()
    ax = axs[1, 1]
    ax.plot(p_d[:, 0], p_d[:, 1], "--", color="black", lw=1.2, label="Desired")
    for label, log in logs.items():
        ax.plot(log["p"][:, 0], log["p"][:, 1], label=label)
    ax.set_title("XY plane"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.axis("equal"); ax.grid(True, ls=":"); ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_wind(log, path):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for i, n in enumerate("xyz"):
        ax.plot(log["t"], log["wind"][:, i], label=f"F{n}")
    ax.set_title("Applied wind force"); ax.set_xlabel("Time [s]"); ax.set_ylabel("Force [N]")
    ax.grid(True, ls=":"); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
