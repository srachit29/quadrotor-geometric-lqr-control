"""
Figure-eight trajectory in NVIDIA Isaac Sim + Pegasus with the Geometric + LQR controller (hover.py).
Final tuning: wn = (2.2, 2.9, 3.5) rad/s, zeta = 1.0. Produces geo_lqr_plot.png and the
SIMULATION ANALYSIS REPORT in the console.
"""
import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation

from examples.rachit.hover import GeometricLQRController, plot_logs, print_summary_report, tune_outer_gains


class PegasusApp:
    def __init__(self, ControllerClass):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        config = MultirotorConfig()
        self.controller = ControllerClass()
        config.backends = [self.controller]
        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(),
            config=config,
        )
        self.world.reset()
        print("🚁 Drone with GeometricLQRController initialized.")
        self.stop_sim = False

    def run(self, max_steps=30000):
        self.timeline.play()
        step_count = 0
        try:
            while simulation_app.is_running() and not self.stop_sim and step_count < max_steps:
                self.world.step(render=True)
                step_count += 1
        except Exception as e:
            print(f"Exception during simulation loop: {e}")
            self.stop_sim = True
        finally:
            self.timeline.stop()


def main():
    controller = GeometricLQRController()

    # --- Final tuned outer-loop gains ---
    wn_x, wn_y, wn_z = 2.2, 2.9, 3.5
    zeta = 1.0
    kx, kv = tune_outer_gains(m=controller.m, wn_x=wn_x, wn_y=wn_y, wn_z=wn_z, zeta=zeta)
    controller.kx = kx
    controller.kv = kv

    # --- Figure-eight with acceleration feedforward ---
    def figure_eight(t):
        x = np.sin(t)
        y = np.sin(t) * np.cos(t)
        z = 1.0 + 0.2 * np.sin(0.5 * t)
        vx = np.cos(t)
        vy = np.cos(2 * t)
        vz = 0.1 * np.cos(0.5 * t)
        ax = -np.sin(t)
        ay = -2 * np.sin(2 * t)
        az = -0.05 * np.sin(0.5 * t)
        yaw = np.sin(t)
        return np.array([x, y, z]), np.array([vx, vy, vz]), yaw, np.array([ax, ay, az])

    controller.set_trajectory(figure_eight)

    app = PegasusApp(lambda: controller)
    app.run(max_steps=50000)

    logs = controller.get_logs()
    print_summary_report(controller, logs, wn_x, wn_y, wn_z, zeta)
    plot_logs(logs)

    simulation_app.close()


if __name__ == "__main__":
    main()
