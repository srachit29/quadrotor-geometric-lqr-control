import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# Dummy class for standalone analysis when simulator is not available
try:
    from pegasus.simulator.logic.state import State
except ImportError:
    print("⚠️ Pegasus State not found. Using dummy class for offline analysis.")
    class State:
        position = np.zeros(3)
        linear_velocity = np.zeros(3)
        attitude = np.array([0,0,0,1])
        angular_velocity = np.zeros(3)

# ================= Controller Backend =================
class GeometricLQRController:
    def __init__(self, config=None, disturbance_enabled=True, wind_model='box'):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.R = Rotation.identity()
        self.w = np.zeros(3)
        self.m = 1.5
        self.g = 9.81
        self.J = np.diag([0.02, 0.02, 0.04])
        self.kx = np.diag([6.0, 6.0, 9.4])
        self.kv = np.diag([5.4, 5.4, 6.8])
        self.Q = np.diag([20.0, 20.0, 12.0, 1.0, 1.0, 1.2])
        self.R_lqr = np.diag([0.05, 0.05, 0.06])
        self.time = 0.0
        self.trajectory_func = self.default_trajectory
        self.input_ref = [0.0, 0.0, 0.0, 0.0]
        self._vehicle = None
        self.received_state = False
        self.disturbance_enabled = disturbance_enabled
        self.wind_model = wind_model

        # Logs
        self.logs = {
            "time": [], "p": [], "p_d": [], "e_p": [],
            "v": [], "v_d": [], "e_v": [],
            "R": [], "R_d": [], "e_R": [],
            "w": [], "w_d": [], "e_w": [],
            "u1": [], "tau": [], "loss": [],
            "F_des": [], "wind_force": []
        }

        self.A = None
        self.B = None
        self.K_lqr = self._compute_lqr_gain(self.J)

        # --- Wind Gust Parameters ---
        self.gust_start = 5.0
        self.gust_end = 15.0  # Extended duration for more interesting models
        self.gust_amplitude = np.array([4.0, 2.0, 0.0])  # N, can be directional

        # Sine model parameters
        self.gust_frequency = np.array([0.5, 0.8, 0.0])  # Hz for x, y, z

        # Dryden turbulence model parameters & state
        self.Lu = 200.0  # Length scale
        self.Lv = 200.0
        self.Lw = 50.0
        self.sigma_u = 1.5
        self.sigma_v = 1.5
        self.sigma_w = 1.0
        self.turbulence_state = np.zeros(3)


    def initialize(self, vehicle):
        """
        This method is called by the Pegasus simulator to link the controller to the vehicle.
        """
        self._vehicle = vehicle

    # Backend stubs
    def reset(self): pass
    def start(self): pass
    def stop(self): pass
    def update_sensor(self, sensor_type=None, data=None): pass
    def update_graphical_sensor(self, sensor_type=None, data=None): pass

    # State update
    def update_state(self, state: State):
        self.p = state.position
        self.v = state.linear_velocity
        self.R = Rotation.from_quat(state.attitude)
        self.w = state.angular_velocity
        self.received_state = True

    def input_reference(self):
        return self.input_ref

    # Main update
    def update(self, dt: float):
        if not self.received_state: return
        self.time += dt

        pos_d, vel_d, yaw_d, acc_d = self.trajectory_func(self.time)

        # Update and get the current wind force based on the selected model
        wind_force = self._update_wind_gust(dt)
        
        # Print to console if wind is active
        if np.linalg.norm(wind_force) > 0.01:
            print(f"💨 Wind Force ({self.wind_model}) at t={self.time:.2f}s: [{wind_force[0]:.2f}, {wind_force[1]:.2f}, {wind_force[2]:.2f}] N")

        e_p = self.p - pos_d
        e_v = self.v - vel_d
        # The desired force must compensate for gravity and counteract the wind disturbance
        F_des = -self.kx @ e_p - self.kv @ e_v + self.m * (acc_d + np.array([0, 0, self.g])) - wind_force
        
        normF = np.linalg.norm(F_des) + 1e-9
        z_b_des = F_des / normF
        x_c = np.array([np.cos(yaw_d), np.sin(yaw_d), 0])
        y_b_des_raw = np.cross(z_b_des, x_c)
        y_b_des = y_b_des_raw / (np.linalg.norm(y_b_des_raw) + 1e-9)
        x_b_des = np.cross(y_b_des, z_b_des)
        R_d = np.column_stack((x_b_des, y_b_des, z_b_des))

        R = self.R.as_matrix()
        e_R_mat = 0.5 * (R_d.T @ R - R.T @ R_d)
        e_R = np.array([-e_R_mat[1, 2], e_R_mat[0, 2], -e_R_mat[0, 1]])
        w_d = np.zeros(3)
        e_w = self.w - w_d

        s = np.hstack((e_R, e_w))
        tau = -self.K_lqr @ s + np.cross(self.w, self.J @ self.w)

        u1 = F_des @ R[:, 2]
        u1 = max(u1, 0.0)
        
        if self._vehicle:
            self.input_ref = self._vehicle.force_and_torques_to_velocities(u1, tau)

        self._log_data(pos_d, vel_d, e_p, e_v, R_d, e_R, w_d, e_w, u1, tau, F_des, wind_force)

    def _update_wind_gust(self, dt: float) -> np.ndarray:
        """
        Calculates the wind force based on the selected model.
        Returns: np.ndarray: The 3D wind force vector in Newtons.
        """
        if not self.disturbance_enabled:
            return np.zeros(3)

        wind_force = np.zeros(3)
        time_in_gust = self.time >= self.gust_start and self.time <= self.gust_end

        if self.wind_model == 'box':
            if time_in_gust:
                wind_force = self.gust_amplitude
        
        elif self.wind_model == 'sine':
            if time_in_gust:
                phases = 2 * np.pi * self.gust_frequency * (self.time - self.gust_start)
                wind_force = self.gust_amplitude * np.sin(phases)
        
        elif self.wind_model == 'ramp':
            if time_in_gust:
                duration = self.gust_end - self.gust_start
                mid_point = self.gust_start + duration / 2.0
                if self.time <= mid_point:
                    scale = (self.time - self.gust_start) / (duration / 2.0)
                    wind_force = self.gust_amplitude * scale
                else:
                    scale = (self.gust_end - self.time) / (duration / 2.0)
                    wind_force = self.gust_amplitude * scale
        
        elif self.wind_model == 'dryden':
            # Simplified Dryden turbulence model using filtered noise
            airspeed = max(1.0, np.linalg.norm(self.v))
            if dt <= 0: return self.turbulence_state # Avoid division by zero if dt is zero

            # Update u, v, w components
            for i, (L, sigma) in enumerate(zip([self.Lu, self.Lv, self.Lw], [self.sigma_u, self.sigma_v, self.sigma_w])):
                coeff = airspeed / L
                noise = np.random.randn() * np.sqrt(2 * sigma**2 * coeff * dt)
                self.turbulence_state[i] += dt * (-coeff * self.turbulence_state[i]) + noise
            
            wind_force = self.turbulence_state.copy()

        return wind_force


    def _log_data(self, p_d, v_d, e_p, e_v, R_d, e_R, w_d, e_w, u1, tau, F_des, wind_force):
        self.logs["time"].append(self.time)
        self.logs["p"].append(self.p.copy()); self.logs["p_d"].append(p_d.copy()); self.logs["e_p"].append(e_p.copy())
        self.logs["v"].append(self.v.copy()); self.logs["v_d"].append(v_d.copy()); self.logs["e_v"].append(e_v.copy())
        self.logs["R"].append(self.R.as_matrix().copy()); self.logs["R_d"].append(R_d.copy()); self.logs["e_R"].append(e_R.copy())
        self.logs["w"].append(self.w.copy()); self.logs["w_d"].append(w_d.copy()); self.logs["e_w"].append(e_w.copy())
        self.logs["u1"].append(u1); self.logs["tau"].append(tau.copy())
        self.logs["loss"].append(e_p.T @ self.kx @ e_p + e_v.T @ self.kv @ e_v + e_R.T @ e_R + e_w.T @ e_w)
        self.logs["F_des"].append(F_des.copy())
        self.logs["wind_force"].append(wind_force.copy())

    def _compute_lqr_gain(self, J):
        Jinv = np.linalg.inv(J)
        self.A = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 3)), np.zeros((3, 3))]])
        self.B = np.vstack([np.zeros((3, 3)), Jinv])
        P = solve_continuous_are(self.A, self.B, self.Q, self.R_lqr)
        return np.linalg.inv(self.R_lqr) @ self.B.T @ P

    def set_trajectory(self, func): self.trajectory_func = func

    def default_trajectory(self, t):
        return np.array([0, 0, 1]), np.zeros(3), 0.0, np.zeros(3)

    def get_logs(self):
        return {k: np.array(v) for k, v in self.logs.items()}

# ================= Utilities =================
def tune_outer_gains(m=1.5, wn_x=2.0, wn_y=2.0, wn_z=2.5, zeta=0.9):
    kx = np.diag([m * wn_x ** 2, m * wn_y ** 2, m * wn_z ** 2])
    kv = np.diag([2 * zeta * m * wn_x, 2 * zeta * m * wn_y, 2 * zeta * m * wn_z])
    print("\n[INFO] Outer-loop gains tuned:")
    print("kx=", np.diag(kx))
    print("kv=", np.diag(kv), "\n")
    return kx, kv

def plot_logs(logs):
    if not logs or len(logs["time"]) < 2: return
    time = logs["time"]
    p_actual = logs["p"]
    p_desired = logs["p_d"]

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs[0, 0].plot(time, p_desired[:, 0], '--', color='orange', label='Desired')
    axs[0, 0].plot(time, p_actual[:, 0], color='blue', label='Actual')
    axs[0, 0].set_title('X Position'); axs[0, 0].legend(); axs[0, 0].grid(True)
    
    axs[0, 1].plot(time, p_desired[:, 1], '--', color='orange', label='Desired')
    axs[0, 1].plot(time, p_actual[:, 1], color='blue', label='Actual')
    axs[0, 1].set_title('Y Position'); axs[0, 1].legend(); axs[0, 1].grid(True)
    
    axs[1, 0].plot(time, p_desired[:, 2], '--', color='orange', label='Desired')
    axs[1, 0].plot(time, p_actual[:, 2], color='blue', label='Actual')
    axs[1, 0].set_title('Z Position'); axs[1, 0].legend(); axs[1, 0].grid(True)
    
    axs[1, 1].plot(p_desired[:, 0], p_desired[:, 1], '--', color='orange', label='Desired')
    axs[1, 1].plot(p_actual[:, 0], p_actual[:, 1], color='blue', label='Actual')
    axs[1, 1].set_title('XY Plane Trajectory'); axs[1, 1].legend(); axs[1, 1].grid(True)
    axs[1, 1].set_xlabel("X [m]"); axs[1, 1].set_ylabel("Y [m]")
    
    plt.tight_layout()
    plt.savefig("figure_eight_trajectory.png", dpi=300)
    print("✅ Figure-eight trajectory plot saved to figure_eight_trajectory.png")

def plot_wind(logs):
    if "wind_force" not in logs or len(logs["wind_force"]) == 0:
        print("⚠️ No wind force data to plot."); return
    wind_force = np.array(logs["wind_force"])
    time = logs["time"]
    plt.figure(figsize=(10, 5))
    plt.plot(time, wind_force[:, 0], label="Fx (Wind)")
    plt.plot(time, wind_force[:, 1], label="Fy (Wind)")
    plt.plot(time, wind_force[:, 2], label="Fz (Wind)")
    plt.title("Applied Wind Force Over Time")
    plt.xlabel("Time [s]"); plt.ylabel("Force [N]")
    plt.legend(); plt.grid(True)
    plt.savefig("wind_force_over_time.png", dpi=300)
    print("✅ Wind force plot saved to wind_force_over_time.png")

def print_summary_logs(controller, logs, **tuning_params):
    if not logs or len(logs["time"]) == 0: return
    e_p = logs["e_p"]
    e_v = logs["e_v"]
    wind = logs["wind_force"]

    print("\n" + "=" * 60)
    print(f"📊 Figure-Eight Simulation Summary (Wind Model: {controller.wind_model})")
    print("=" * 60)
    print(f"Outer-loop gains: kx={np.diag(controller.kx)}, kv={np.diag(controller.kv)}")
    print(f"LQR Q={np.diag(controller.Q)}, R={np.diag(controller.R_lqr)}")
    print(f"Position RMSE: {np.sqrt(np.mean(np.sum(e_p**2, axis=1))):.4f} m")
    print(f"Velocity RMSE: {np.sqrt(np.mean(np.sum(e_v**2, axis=1))):.4f} m/s")
    if len(wind) > 0 and np.any(wind):
        max_wind = np.max(np.abs(wind), axis=0)
        print(f"Max Absolute Wind Force Applied: [{max_wind[0]:.2f}, {max_wind[1]:.2f}, {max_wind[2]:.2f}] N")
    print("=" * 60)
