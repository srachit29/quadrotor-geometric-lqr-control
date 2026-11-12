| Experiment | Position RMSE [m] | Velocity RMSE [m/s] | Attitude RMSE [rad] | Max roll/pitch error [rad] | Max thrust [N] | Max torque x/y/z [N m] |
|---|---|---|---|---|---|---|
| Hover at 1 m (Geometric + LQR) | 0.196 | 0.246 | 0.000 | 0.000 | 25.7 | 0.00/0.00/0.00 |
| Figure-eight, LQR only | 0.331 | 0.923 | 0.396 | 0.620 | 26.7 | 2.45/2.45/0.23 |
| Figure-eight, Geometric + LQR, aggressive tuning (wn 3.0/3.2/4.0) | 0.151 | 0.336 | 0.247 | 0.394 | 38.2 | 4.81/4.38/0.38 |
| Figure-eight, Geometric + LQR, final tuning (wn 2.2/2.9/3.5) | 0.167 | 0.328 | 0.241 | 0.279 | 32.9 | 5.07/3.75/0.35 |
| Large figure-eight with sinusoidal wind, Geometric + LQR (wn 3.0/3.2/4.0) | 0.152 | 0.536 | 0.309 | 0.459 | 17.2 | 5.21/9.01/10.59 |
