# Isaac Sim + Pegasus scripts

Original project code (September 2025). The scripts fly the Iris quadrotor in NVIDIA Isaac Sim with the [Pegasus Simulator](https://github.com/PegasusSimulator/PegasusSimulator), using `GeometricLQRController` as a custom Pegasus backend.

| File | What it does |
|---|---|
| `hover.py` | Final Geometric + LQR controller backend, plots, and the full `SIMULATION ANALYSIS REPORT` |
| `compare.py` | Same controller with wind disturbance models: `box`, `sine`, `ramp`, `dryden` |
| `run_hover.py` | Hover at (0, 0, 1) m, ωn = 2.6 / 2.6 / 2.8 rad/s |
| `run_figure_eight.py` | Figure-eight (x = sin t, y = sin t cos t), final tuning ωn = 2.2 / 2.9 / 3.5 rad/s |
| `run_figure_eight_wind.py` | Larger figure-eight (2 m × 1.5 m, 20 s period) with sinusoidal wind, ωn = 3.0 / 3.2 / 4.0 rad/s |

## Run

1. Install Isaac Sim and the Pegasus Simulator (see the Pegasus documentation).
2. Copy the files into `PegasusSimulator/examples/rachit/` (the run scripts import `examples.rachit.hover` and `examples.rachit.compare`).
3. From the PegasusSimulator folder, run a script with Isaac Sim's Python, for example:

```bash
ISAACSIM_PYTHON examples/rachit/run_figure_eight.py
```

Plots are saved in the working folder after the simulation window closes.

## Notes

- **Wind model.** In `compare.py` the wind force is injected at the controller's force command (`F_des − F_wind`), so it acts as a force disturbance on the vehicle without needing a Pegasus wind plugin. The sine model applies 4 N at 0.5 Hz in X and 2 N at 0.8 Hz in Y between t = 5 s and 15 s.
- **Starting point.** In the hover and figure-eight runs the drone spawns on the floor (z = 0.07 m), so the takeoff is part of every run and dominates the maximum Z error. The wind run spawns at the trajectory's start point.
- **LQR-only baseline.** The single 12-state LQR used for comparison is not in this folder; it is reimplemented in `../python_sim/controllers.py` with the same Q and R weights.
- **Small fixes vs. the original files:** `run_hover.py` calls `print_summary_report` (the name used in the final `hover.py`) and sets the hover set-point explicitly; `run_figure_eight.py` was rebuilt from the original figure-eight script to import the final `hover.py`.
