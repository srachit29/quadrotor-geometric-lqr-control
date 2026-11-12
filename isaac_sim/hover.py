import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.linalg import solve_continuous_are

# Note: The following imports are placeholders. You need the full Pegasus/IsaacSim environment to run.
try:
    from pegasus.simulator.logic.backends import Backend
    from pegasus.simulator.logic.state import State
except ImportError:
    print("Warning: Pegasus/IsaacSim libraries not found. Using dummy classes for standalone execution.")
    class Backend: pass
    class State: pass

# ============= Controller Definition ==================
class GeometricLQRController(Backend):
    def __init__(self):
        # Vehicle & World Parameters
        self.p = np.zeros(3); self.v = np.zeros(3)
        self.R = Rotation.identity(); self.w = np.zeros(3)
        self.m = 1.5; self.g = 9.81
        self.J = np.diag([0.02, 0.02, 0.04])
        
        # Controller Gains & LQR Weights
        self.kx = np.diag([6.0, 6.0, 9.4])
        self.kv = np.diag([5.4, 5.4, 6.8])
        self.Q = np.diag([20.0, 20.0, 12.0, 1.0, 1.0, 1.2])
        self.R_lqr = np.diag([0.05, 0.05, 0.06])
        self.K_lqr = self._compute_lqr_gain()
        
        # Simulation State & Logging
        self.time = 0.0
        self.trajectory_func = lambda t: (np.zeros(3),np.zeros(3),0.,np.zeros(3))
        self.input_ref = [0.0, 0.0, 0.0, 0.0]
        self._vehicle = None; self.received_state = False
        
        self.logs = {
            "time": [], "p": [], "p_d": [], "e_p": [], "v": [], "v_d": [], "e_v": [],
            "R": [], "R_d": [], "e_R": [], "w": [], "w_d": [], "e_w": [],
            "u1": [], "tau": [], "F_des": []
        }
        print("[INFO] LQR gain matrix K computed:\n", self.K_lqr)

    def _compute_lqr_gain(self):
        J_inv = np.linalg.inv(self.J)
        A = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 3)), np.zeros((3, 3))]])
        B = np.vstack([np.zeros((3, 3)), J_inv])
        P = solve_continuous_are(A, B, self.Q, self.R_lqr)
        K = np.linalg.inv(self.R_lqr) @ B.T @ P
        return K

    def update(self, dt: float):
        if not self.received_state or self._vehicle is None: return
        self.time += dt
        
        traj = self.trajectory_func(self.time)
        p_d, v_d, yaw_d = traj[:3]
        a_d = traj[3] if len(traj) == 4 else np.zeros(3)
        
        e_p = self.p - p_d
        e_v = self.v - v_d
        F_des = -self.kx @ e_p - self.kv @ e_v + self.m * (a_d + np.array([0, 0, self.g]))
        
        normF = np.linalg.norm(F_des) + 1e-9; z_b_des = F_des / normF
        x_c = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])
        norm_y_raw = np.linalg.norm(np.cross(z_b_des, x_c)) + 1e-9
        y_b_des = np.cross(z_b_des, x_c) / norm_y_raw
        x_b_des = np.cross(y_b_des, z_b_des)
        R_d = np.column_stack((x_b_des, y_b_des, z_b_des))
        
        R_matrix = self.R.as_matrix()
        e_R = 0.5 * self.vee(R_d.T @ R_matrix - R_matrix.T @ R_d)
        w_d = np.zeros(3)
        e_w = self.w - w_d
        s = np.hstack((e_R, e_w))
        
        tau = -self.K_lqr @ s + np.cross(self.w, self.J @ self.w)
        
        u1 = F_des @ R_matrix[:, 2]
        u1 = max(u1, 0.0)
        
        if self._vehicle: self.input_ref = self._vehicle.force_and_torques_to_velocities(u1, tau)
        self._log_data(p_d,v_d,e_p,e_v,R_d,e_R,w_d,e_w,u1,tau,F_des)

    def _log_data(self,p_d,v_d,e_p,e_v,R_d,e_R,w_d,e_w,u1,tau,F_des):
        self.logs["time"].append(self.time)
        self.logs["p"].append(self.p.copy()); self.logs["p_d"].append(p_d.copy()); self.logs["e_p"].append(e_p.copy())
        self.logs["v"].append(self.v.copy()); self.logs["v_d"].append(v_d.copy()); self.logs["e_v"].append(e_v.copy())
        self.logs["R"].append(self.R.as_matrix().copy()); self.logs["R_d"].append(R_d.copy()); self.logs["e_R"].append(e_R.copy())
        self.logs["w"].append(self.w.copy()); self.logs["w_d"].append(w_d.copy()); self.logs["e_w"].append(e_w.copy())
        self.logs["u1"].append(u1); self.logs["tau"].append(tau.copy())
        self.logs["F_des"].append(F_des.copy())

    def update_state(self, state: State):
        self.p = state.position; self.v = state.linear_velocity
        self.R = Rotation.from_quat(state.attitude); self.w = state.angular_velocity
        self.received_state = True
        
    @staticmethod
    def vee(S): return np.array([-S[1, 2], S[0, 2], -S[0, 1]])
    def set_trajectory(self, func): self.trajectory_func = func
    def get_logs(self): return {k: np.array(v) for k, v in self.logs.items()}
    def input_reference(self): return self.input_ref
    def reset(self): pass
    def start(self): pass
    def stop(self): pass
    def update_sensor(self, sensor_type: str, data): pass
    def update_graphical_sensor(self, sensor_type: str, data): pass

# ============= Helper, Plotting & Analysis Functions ==================
def tune_outer_gains(m=1.5, wn_x=2.0, wn_y=2.0, wn_z=2.5, zeta=0.9):
    kx = np.diag([m*wn_x**2, m*wn_y**2, m*wn_z**2])
    kv = np.diag([2*zeta*m*wn_x, 2*zeta*m*wn_y, 2*zeta*m*wn_z])
    return kx, kv

def plot_logs(logs):
    if not logs or len(logs["time"]) < 2: return
    time=logs["time"]; p_actual=logs["p"]; p_desired=logs["p_d"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Trajectory Tracking Performance (Geo+LQR Controller)', fontsize=16)
    axs[0,0].plot(time, p_desired[:,0], '--', c='orange', label='Desired')
    axs[0,0].plot(time, p_actual[:,0], c='blue', label='Geo+LQR')
    axs[0,0].set_title('X Position'); axs[0,0].legend(); axs[0,0].grid(True,ls=':')
    axs[0,1].plot(time, p_desired[:,1], '--', c='orange', label='Desired')
    axs[0,1].plot(time, p_actual[:,1], c='blue', label='Geo+LQR')
    axs[0,1].set_title('Y Position'); axs[0,1].legend(); axs[0,1].grid(True,ls=':')
    axs[1,0].plot(time, p_desired[:,2], '--', c='orange', label='Desired')
    axs[1,0].plot(time, p_actual[:,2], c='blue', label='Geo+LQR')
    axs[1,0].set_title('Z Position'); axs[1,0].legend(); axs[1,0].grid(True,ls=':')
    axs[1,1].plot(p_desired[:,0],p_desired[:,1], '--', c='orange', label='Desired')
    axs[1,1].plot(p_actual[:,0],p_actual[:,1], c='blue', label='Geo+LQR')
    axs[1,1].set_title('XY Plane'); axs[1,1].legend(); axs[1,1].grid(True,ls=':')
    axs[1,1].set_aspect('equal', 'box')
    plt.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig("geo_lqr_plot.png", dpi=300)
    print("\n✅ Geo+LQR plot saved to geo_lqr_plot.png")

def print_summary_report(controller, logs, wn_x, wn_y, wn_z, zeta):
    if not logs or len(logs["time"]) < 2:
        print("Logs are empty. Cannot generate report.")
        return
    e_p=logs["e_p"]; rmse_p=np.sqrt(np.mean(np.sum(e_p**2,axis=1))); max_abs_e_p=np.max(np.abs(e_p),axis=0)
    e_v=logs["e_v"]; rmse_v=np.sqrt(np.mean(np.sum(e_v**2,axis=1))); max_abs_e_v=np.max(np.abs(e_v),axis=0)
    e_R=logs["e_R"]; rmse_R=np.sqrt(np.mean(np.sum(e_R**2,axis=1))); max_abs_e_R=np.max(np.abs(e_R),axis=0)
    e_w=logs["e_w"]; rmse_w=np.sqrt(np.mean(np.sum(e_w**2,axis=1))); max_abs_e_w=np.max(np.abs(e_w),axis=0)
    mean_u1=np.mean(logs["u1"]); max_u1=np.max(logs["u1"]); hover_thrust=controller.m*controller.g
    mean_abs_tau=np.mean(np.abs(logs["tau"]),axis=0); max_abs_tau=np.max(np.abs(logs["tau"]),axis=0)

    print("\n\n" + "="*70); print("📊 SIMULATION ANALYSIS REPORT"); print("="*70 + "\n")
    print("[CONTROLLER PARAMETERS]"); print(f"- Vehicle Mass (m): {controller.m} kg"); print(f"- Vehicle Inertia (J): {np.diag(controller.J)}\n")
    print("[1. OUTER-LOOP TRANSLATIONAL CONTROL]")
    print("- Tuning Parameters:"); print(f"  - Natural Frequencies (wn): [{wn_x:.3f} {wn_y:.3f} {wn_z:.3f}] rad/s")
    print("- Equations:"); print("  - kx = m * wn^2"); print("  - kv = 2 * zeta * m * wn")
    print("- Resulting Gains:"); print(f"  - Proportional (kx): {np.diag(controller.kx)}"); print(f"  - Derivative   (kv): {np.diag(controller.kv)}\n")
    print("[2. INNER-LOOP ATTITUDE CONTROL (LQR)]")
    print("- LQR Weighting Matrices:"); print(f"  - State (Q):   {np.diag(controller.Q)}"); print(f"  - Control (R): {np.diag(controller.R_lqr)}")
    print("- Resulting LQR Gain (K_lqr):")
    for row in controller.K_lqr: print(f"  [{' '.join([f'{elem:8.4f}' for elem in row])}]")
    print("")
    print("[3. PERFORMANCE METRICS & ERRORS]"); print("- Position Tracking:")
    print("  - Equation: e_p = p_actual - p_desired"); print(f"  - RMSE: {rmse_p:.4f} m"); print(f"  - Max Absolute Error (X,Y,Z): [{max_abs_e_p[0]:.3f} {max_abs_e_p[1]:.3f} {max_abs_e_p[2]:.3f}] m")
    print("- Velocity Tracking:"); print("  - Equation: e_v = v_actual - v_desired"); print(f"  - RMSE: {rmse_v:.4f} m/s"); print(f"  - Max Absolute Error (Vx,Vy,Vz): [{max_abs_e_v[0]:.3f} {max_abs_e_v[1]:.3f} {max_abs_e_v[2]:.3f}] m/s")
    print("- Attitude Tracking (SO(3) vee map):"); print("  - Equation: e_R = 0.5 * (R_d^T * R - R^T * R_d)^vee"); print(f"  - RMSE: {rmse_R:.4f} rad"); print(f"  - Max Absolute Error (rol,pit,yaw): [{max_abs_e_R[0]:.3f} {max_abs_e_R[1]:.3f} {max_abs_e_R[2]:.3f}] rad")
    print("- Angular Velocity Tracking:"); print("  - Equation: e_w = w_actual - w_desired"); print(f"  - RMSE: {rmse_w:.4f} rad/s"); print(f"  - Max Absolute Error (wx,wy,wz): [{max_abs_e_w[0]:.3f} {max_abs_e_w[1]:.3f} {max_abs_e_w[2]:.3f}] rad/s\n")
    print("[4. CONTROL EFFORT & DISTURBANCE]"); print("- Total Thrust (u1):")
    print("  - Equation: u1 = F_des .dot. R_z"); print(f"  - Mean: {mean_u1:.3f} N (Hover thrust is approx. {hover_thrust:.3f} N)"); print(f"  - Max:  {max_u1:.3f} N")
    print("- Control Torques (tau):"); print("  - Equation: tau = -K_lqr * [e_R, e_w]^T + w x (Jw)")
    print(f"  - Mean Absolute (tx,ty,tz): [{mean_abs_tau[0]:.3f} {mean_abs_tau[1]:.3f} {mean_abs_tau[2]:.3f}] Nm"); print(f"  - Max Absolute  (tx,ty,tz): [{max_abs_tau[0]:.3f} {max_abs_tau[1]:.3f} {max_abs_tau[2]:.3f}] Nm")
    print("- Wind Disturbance:"); print("  - Model: None"); print("  - Max Absolute Force Applied: [0.000 0.000 0.000] N")
    print("="*70)