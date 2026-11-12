import numpy as np
from scipy.spatial.transform import Rotation

# Pegasus/IsaacSim imports
try:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})
    import omni.timeline
    from omni.isaac.core.world import World
    from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
    from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
except ImportError:
    print("⚠️ Pegasus/IsaacSim not found. Cannot run simulation.")
    # Define dummy classes to avoid crashing if imports fail
    class SimulationApp:
        def __init__(self, config=None): pass
        def is_running(self): return False
        def close(self): pass
    simulation_app = SimulationApp()
    class World:
        def __init__(self, **kwargs): pass
        def step(self, render=False): pass
        def reset(self): pass
    class PegasusInterface:
        def __init__(self): self._world_settings = {}; self.world = World()
        def load_environment(self, env): pass
    class Multirotor:
        def __init__(self, *args, **kwargs): pass
    class MultirotorConfig:
        def __init__(self): self.backends = []
    omni = type('omni', (), {'timeline': type('timeline', (), {'get_timeline_interface': lambda: type('timeline_interface', (), {'play': lambda: None, 'stop': lambda: None})()})()})
    ROBOTS = {"Iris": None}
    SIMULATION_ENVIRONMENTS = {"Curved Gridroom": None}


# Import the controller and utility functions from the backend script
from examples.rachit.compare import GeometricLQRController, plot_logs, plot_wind, print_summary_logs, tune_outer_gains

# ================= Pegasus App Wrapper =================
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
        
        # Get the starting position from the trajectory function at t=0
        start_pos, _, _, _ = self.controller.trajectory_func(0)

        # Spawn drone at the trajectory's starting position
        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            start_pos,
            Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(),
            config=config,
        )
        self.world.reset()
        print(f"🚁 Drone with Geometric LQR controller initialized at {start_pos}.")
        self.stop_sim = False

    def run(self, max_steps=30000):
        self.timeline.play()
        step_count = 0
        try:
            while simulation_app.is_running() and not self.stop_sim and step_count < max_steps:
                self.world.step(render=True)
                step_count += 1
                if step_count % 1000 == 0:
                    print(f"Sim step: {step_count}/{max_steps}")
        except Exception as e:
            print(f"Exception during simulation loop: {e}")
            self.stop_sim = True
        finally:
            self.timeline.stop()
            print("Simulation stopped.")

# ================= Main Execution =================
def main():
    # --- CHOOSE YOUR WIND MODEL ---
    # Options: 'box', 'sine', 'ramp', 'dryden'
    WIND_CHOICE = 'sine' 
    
    # Instantiate the controller with the selected wind model
    controller = GeometricLQRController(disturbance_enabled=True, wind_model=WIND_CHOICE)
    print(f"💨 Running simulation with '{WIND_CHOICE}' wind model.")

    # --- Final Tuned Gains for Figure-Eight (More Aggressive) ---
    wn_x, wn_y, wn_z = 3.0, 3.2, 4.0
    zeta = 1.0

    kx, kv = tune_outer_gains(m=controller.m, wn_x=wn_x, wn_y=wn_y, wn_z=wn_z, zeta=zeta)
    controller.kx = kx
    controller.kv = kv
    
    # --- Figure-Eight Trajectory with Acceleration Feedforward ---
    def figure_eight_trajectory(t):
        T = 20.0 # Period of one full figure-eight
        w = 2 * np.pi / T
        
        # Position
        x = 2.0 * np.sin(w * t)
        y = 1.5 * np.sin(2 * w * t)
        z = 1.5 + 0.5 * np.sin(w*t + np.pi/2) # Bob up and down
        
        # Velocity (1st derivative)
        vx = 2.0 * w * np.cos(w * t)
        vy = 1.5 * 2 * w * np.cos(2 * w * t)
        vz = 0.5 * w * np.cos(w*t + np.pi/2)
        
        # Acceleration (2nd derivative)
        ax = -2.0 * w**2 * np.sin(w * t)
        ay = -1.5 * 4 * w**2 * np.sin(2 * w * t)
        az = -0.5 * w**2 * np.sin(w*t + np.pi/2)
        
        # Yaw can be set to a constant or made to follow the path
        yaw = np.arctan2(vy, vx)
        
        return np.array([x, y, z]), np.array([vx, vy, vz]), yaw, np.array([ax, ay, az])

    controller.set_trajectory(figure_eight_trajectory)

    # --- Run Pegasus Simulation ---
    # The lambda ensures the latest controller instance is passed
    app = PegasusApp(lambda: controller)
    app.run(max_steps=60000) 

    # --- Logging & Analysis ---
    logs = controller.get_logs()
    print_summary_logs(controller, logs, wn_x=wn_x, wn_y=wn_y, wn_z=wn_z)
    plot_logs(logs)
    plot_wind(logs)

    simulation_app.close()

if __name__ == "__main__":
    main()
