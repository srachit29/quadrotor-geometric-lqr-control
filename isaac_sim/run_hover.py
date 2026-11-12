import numpy as np
from isaacsim import SimulationApp

# Launch Isaac Sim
simulation_app = SimulationApp({"headless": False})

import omni.timeline
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation

# Import from the re-tuned backend script
from examples.rachit.hover import (
    GeometricLQRController,
    plot_logs,
    print_summary_report,
    tune_outer_gains,
)


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
        print("🚁 Drone with FINAL TUNED Geometric LQR controller initialized.")
        self.stop_sim = False

    def run(self, max_steps=30000):
        self.timeline.play()
        step_count = 0
        try:
            while simulation_app.is_running() and not self.stop_sim and step_count < max_steps:
                self.world.step(render=True)
                step_count += 1
        finally:
            self.timeline.stop()


def main():
    controller = GeometricLQRController()

    # --- ✅ FINAL TUNING: A balanced, critically damped response for ACCURACY ---
    wn_x = 2.6
    wn_y = 2.6
    wn_z = 2.8
    # Critical damping for a fast response without overshoot
    zeta = 1.0

    kx, kv = tune_outer_gains(m=controller.m, wn_x=wn_x, wn_y=wn_y, wn_z=wn_z, zeta=zeta)
    controller.kx = kx
    controller.kv = kv

    # Hover set-point: (0, 0, 1) m, yaw 0
    controller.set_trajectory(lambda t: (np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.0, np.zeros(3)))

    # Assign controller to app
    app = PegasusApp(lambda: controller)

    # Run simulation
    app.run(max_steps=30000)

    # Logging & Analysis
    logs = controller.get_logs()
    print_summary_report(controller, logs, wn_x, wn_y, wn_z, zeta)
    plot_logs(logs)

    # Close simulator
    simulation_app.close()


if __name__ == "__main__":
    main()
