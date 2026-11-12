"""Controllers compared in this project.

1. GeometricLQRController: geometric PD position loop + LQR attitude loop on SO(3)
   (same equations as isaac_sim/hover.py).
2. LQROnlyController: one LQR for the full 12-state model, linearised at hover.
"""
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.spatial.transform import Rotation

G = 9.81
E3 = np.array([0.0, 0.0, 1.0])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def outer_gains(m, wn, zeta=1.0):
    """Pole placement: kx = m*wn^2, kv = 2*zeta*m*wn (per axis)."""
    wn = np.asarray(wn, dtype=float)
    return np.diag(m * wn**2), np.diag(2 * zeta * m * wn)


def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


class GeometricLQRController:
    name = "Geometric + LQR"

    def __init__(self, m=1.5, J=(0.02, 0.02, 0.04), wn=(2.2, 2.9, 3.5), zeta=1.0,
                 Q=(20.0, 20.0, 12.0, 1.0, 1.0, 1.2), R=(0.05, 0.05, 0.06)):
        self.m = m
        self.J = np.diag(J)
        self.wn = np.asarray(wn, dtype=float)
        self.kx, self.kv = outer_gains(m, wn, zeta)
        self.Q = np.diag(Q)
        self.R_lqr = np.diag(R)
        # Linearised attitude-error dynamics: d/dt [e_R, e_w] = A [e_R, e_w] + B tau
        A = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 3)), np.zeros((3, 3))]])
        B = np.vstack([np.zeros((3, 3)), np.linalg.inv(self.J)])
        self.K_lqr = lqr(A, B, self.Q, self.R_lqr)

    def __call__(self, p, v, R, w, p_d, v_d, yaw_d, a_d):
        # Outer loop: desired force (PD + gravity compensation + acceleration feedforward)
        e_p, e_v = p - p_d, v - v_d
        F_des = -self.kx @ e_p - self.kv @ e_v + self.m * (a_d + G * E3)

        # Desired attitude: body z along F_des, heading from yaw_d
        z_b = F_des / (np.linalg.norm(F_des) + 1e-9)
        x_c = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])
        y_b = np.cross(z_b, x_c)
        y_b /= np.linalg.norm(y_b) + 1e-9
        x_b = np.cross(y_b, z_b)
        R_d = np.column_stack((x_b, y_b, z_b))

        # Inner loop: LQR on the SO(3) attitude error
        e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        e_w = w
        tau = -self.K_lqr @ np.hstack((e_R, e_w)) + np.cross(w, self.J @ w)

        u1 = max(F_des @ R[:, 2], 0.0)
        return u1, tau, {"e_R": e_R, "e_w": e_w}


class LQROnlyController:
    """Single LQR on x = [p, v, (roll, pitch, yaw), w], linearised at hover."""
    name = "LQR only"

    def __init__(self, m=1.5, J=(0.02, 0.02, 0.04),
                 Q=(10, 10, 15, 0.1, 0.1, 0.1, 5, 5, 8, 0.1, 0.1, 0.1),
                 R=(0.1, 0.5, 0.5, 0.5)):
        self.m = m
        self.J = np.diag(J)
        Ji = np.linalg.inv(self.J)
        A = np.zeros((12, 12))
        A[0:3, 3:6] = np.eye(3)               # p_dot = v
        A[3, 7] = G                           # x_ddot =  g * pitch
        A[4, 6] = -G                          # y_ddot = -g * roll
        A[6:9, 9:12] = np.eye(3)              # angles_dot = w (small angles)
        B = np.zeros((12, 4))
        B[5, 0] = 1.0 / m                     # z_ddot = du1 / m
        B[9:12, 1:4] = Ji                     # w_dot = J^-1 tau
        self.Q = np.diag(Q)
        self.R_lqr = np.diag(R)
        self.K = lqr(A, B, self.Q, self.R_lqr)

    def __call__(self, p, v, R, w, p_d, v_d, yaw_d, a_d):
        roll, pitch, yaw = Rotation.from_matrix(R).as_euler("xyz")
        yaw_err = (yaw - yaw_d + np.pi) % (2 * np.pi) - np.pi
        x_err = np.hstack((p - p_d, v - v_d, [roll, pitch, yaw_err], w))
        u = -self.K @ x_err
        u1 = max(self.m * G + u[0], 0.0)
        tau = u[1:4]
        return u1, tau, {"e_R": np.array([roll, pitch, yaw_err]), "e_w": w}
