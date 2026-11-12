"""Rigid-body quadrotor model (same mass and inertia as the Isaac Sim / Pegasus setup).

State: position p, velocity v (world frame), rotation R (body -> world), body rates w.
Input: collective thrust u1 [N] along body z, body torques tau [N m].
External: wind force F_wind [N] in the world frame.
"""
import numpy as np
from scipy.spatial.transform import Rotation

G = 9.81
E3 = np.array([0.0, 0.0, 1.0])


class Quadrotor:
    def __init__(self, m=1.5, J=(0.02, 0.02, 0.04), p0=(0.0, 0.0, 0.07)):
        self.m = m
        self.J = np.diag(J)
        self.J_inv = np.linalg.inv(self.J)
        self.p = np.array(p0, dtype=float)
        self.v = np.zeros(3)
        self.R = np.eye(3)
        self.w = np.zeros(3)

    def step(self, u1, tau, dt, F_wind=np.zeros(3)):
        """Semi-implicit Euler step; rotation updated on SO(3) with the exponential map."""
        acc = (u1 * self.R @ E3 + F_wind) / self.m - G * E3
        w_dot = self.J_inv @ (tau - np.cross(self.w, self.J @ self.w))

        self.v = self.v + acc * dt
        self.p = self.p + self.v * dt
        self.w = self.w + w_dot * dt
        self.R = self.R @ Rotation.from_rotvec(self.w * dt).as_matrix()

        # Simple ground contact: the drone starts on the floor (z = 0.07 m, as in Pegasus)
        if self.p[2] < 0.0:
            self.p[2] = 0.0
            self.v[2] = max(self.v[2], 0.0)
