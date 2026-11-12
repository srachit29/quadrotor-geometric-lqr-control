"""Run every experiment from the project and save plots + a results table to ../results.

    python run_experiments.py
"""
import os
import numpy as np

from controllers import GeometricLQRController, LQROnlyController
from scenarios import hover, figure_eight, figure_eight_large, no_wind, sine_wind
from simulate import run, metrics, print_report, plot_tracking, plot_wind

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUT, exist_ok=True)

FLOOR = (0.0, 0.0, 0.07)  # Pegasus spawn height

EXPERIMENTS = [
    # (key, title, controller, trajectory, wind, duration, start position)
    ("hover", "Hover at 1 m (Geometric + LQR)", GeometricLQRController(wn=(2.6, 2.6, 2.8)), hover, no_wind, 10.0, FLOOR),
    ("lqr_only", "Figure-eight, LQR only", LQROnlyController(), figure_eight, no_wind, 16.0, FLOOR),
    ("geometric", "Figure-eight, Geometric + LQR, aggressive tuning (wn 3.0/3.2/4.0)",
     GeometricLQRController(wn=(3.0, 3.2, 4.0)), figure_eight, no_wind, 16.0, FLOOR),
    ("geo_lqr", "Figure-eight, Geometric + LQR, final tuning (wn 2.2/2.9/3.5)",
     GeometricLQRController(wn=(2.2, 2.9, 3.5)), figure_eight, no_wind, 16.0, FLOOR),
    ("geo_lqr_wind", "Large figure-eight with sinusoidal wind, Geometric + LQR (wn 3.0/3.2/4.0)",
     GeometricLQRController(wn=(3.0, 3.2, 4.0)), figure_eight_large, sine_wind, 25.0, (0.0, 0.0, 2.0)),
]


def main():
    logs, rows = {}, []
    for key, title, ctrl, traj, wind, T, p0 in EXPERIMENTS:
        log = run(ctrl, traj, wind, T=T, p0=p0)
        m = metrics(log)
        logs[key] = log
        print_report(title, ctrl, m)
        rows.append((title, m))
        plot_tracking({ctrl.name: log}, title, os.path.join(OUT, f"{key}.png"))

    plot_tracking({"LQR only": logs["lqr_only"], "Geometric + LQR": logs["geo_lqr"]},
                  "Figure-eight tracking: LQR only vs. Geometric + LQR", os.path.join(OUT, "comparison.png"))
    plot_wind(logs["geo_lqr_wind"], os.path.join(OUT, "wind_force.png"))

    with open(os.path.join(OUT, "results.md"), "w", encoding="utf-8") as f:
        f.write("| Experiment | Position RMSE [m] | Velocity RMSE [m/s] | Attitude RMSE [rad] | "
                "Max roll/pitch error [rad] | Max thrust [N] | Max torque x/y/z [N m] |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for title, m in rows:
            tq = "/".join(f"{x:.2f}" for x in m["max_torque"])
            f.write(f"| {title} | {m['pos_rmse']:.3f} | {m['vel_rmse']:.3f} | {m['att_rmse']:.3f} | "
                    f"{m['max_att_err']:.3f} | {m['max_thrust']:.1f} | {tq} |\n")
    print(f"\nPlots and results.md saved to {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
