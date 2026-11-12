"""Reference trajectories and wind models (same as in the Isaac Sim experiments)."""
import numpy as np


def hover(t):
    return np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.0, np.zeros(3)


def figure_eight(t):
    """Figure-eight: x = sin t, y = sin t cos t, z = 1 + 0.2 sin(t/2), yaw = sin t."""
    p = np.array([np.sin(t), np.sin(t) * np.cos(t), 1.0 + 0.2 * np.sin(0.5 * t)])
    v = np.array([np.cos(t), np.cos(2 * t), 0.1 * np.cos(0.5 * t)])
    a = np.array([-np.sin(t), -2 * np.sin(2 * t), -0.05 * np.sin(0.5 * t)])
    return p, v, np.sin(t), a


def figure_eight_large(t, T=20.0):
    """Larger, slower figure-eight used in the wind test: 2 m x 1.5 m, 20 s period, yaw along the path."""
    w = 2 * np.pi / T
    p = np.array([2.0 * np.sin(w * t), 1.5 * np.sin(2 * w * t), 1.5 + 0.5 * np.sin(w * t + np.pi / 2)])
    v = np.array([2.0 * w * np.cos(w * t), 3.0 * w * np.cos(2 * w * t), 0.5 * w * np.cos(w * t + np.pi / 2)])
    a = np.array([-2.0 * w**2 * np.sin(w * t), -6.0 * w**2 * np.sin(2 * w * t), -0.5 * w**2 * np.sin(w * t + np.pi / 2)])
    return p, v, np.arctan2(v[1], v[0]), a


def no_wind(t):
    return np.zeros(3)


def sine_wind(t, start=5.0, end=15.0, amp=(4.0, 2.0, 0.0), freq=(0.5, 0.8, 0.0)):
    """Sinusoidal gusts between 5 s and 15 s: 4 N peak at 0.5 Hz in X, 2 N peak at 0.8 Hz in Y."""
    if not start <= t <= end:
        return np.zeros(3)
    return np.asarray(amp) * np.sin(2 * np.pi * np.asarray(freq) * (t - start))
