
from typing import Union, Dict, List, Tuple, Set

from simulation_environment import utils
from simulation_environment.vehicle.behavior import IDMVehicle
import numpy as np
from utils.cubic_spline import Spline2D
import logger
log = logger.get_logger(__name__)


class HumanLikeVehicle(IDMVehicle):
    """
    Create a human-like (IRL) driving agent.
    """
    TAU_A = 0.2  # [s]
    TAU_DS = 0.1  # [s]
    PURSUIT_TAU = 1.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 0.2  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    MAX_VELOCITY = 30  # [m/s]

    def __init__(self,
                 road,
                 name,
                 position,
                 heading=0,
                 velocity=0,
                 target_lane_index=None,
                 target_velocity=15,  # Speed reference
                 route=None,
                 timer=None,
                 start_step=0,
                 v_length=None,
                 v_width=None,
                 ngsim_traj=None,
                 ):
        super(HumanLikeVehicle, self).__init__(road, name, position, heading, velocity, target_lane_index, target_velocity, route, timer)

        self.start_step = start_step
        self.ngsim_traj = ngsim_traj
        self.traj = np.array(self.position)
        self.sim_steps = 1
        self.planned_trajectory = ngsim_traj[:, :2]
        self.planned_speed = ngsim_traj[:, 2:3]
        self.planned_heading = ngsim_traj[:, 3:4]

        self.total_traj_spline = None
        self.build_trajs_spline()

        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.action_history = []
        self.steering_noise = None
        self.LENGTH = v_length  # Vehicle length [m]
        self.WIDTH = v_width  # Vehicle width [m]

    def act(self, action: Union[dict, str, int] = None):

        try:
            control_heading, acceleration = self.control_vehicle(self.planned_trajectory[self.sim_steps],
                                                                 self.planned_speed[self.sim_steps],
                                                                 self.planned_heading[self.sim_steps])
            self.action = {'steering': control_heading, 'acceleration': acceleration}
        except Exception as e:
            log.error(f'Error: {e}')
            raise ValueError(f'Invalid action, error {e}')

    def control_vehicle(self, next_position, next_speed, next_heading):

        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(next_heading - self.heading)

        # Heading rate to steering angle
        steering_angle = np.arctan(self.LENGTH / utils.not_zero(self.speed) * heading_rate_command)
        # steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        acceleration = 10 * (next_speed - self.speed)
        return steering_angle, acceleration

    def step(self, dt):

        super(HumanLikeVehicle, self).step(dt)
        if self.sim_steps < len(self.planned_trajectory):
            self.heading = self.planned_heading[self.sim_steps].squeeze()
            self.speed = self.planned_speed[self.sim_steps].squeeze()
            self.position = self.planned_trajectory[self.sim_steps]

        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.speed)
        self.crash_history.append(self.crashed)
        self.action_history.append(self.action)
        self.traj = np.append(self.traj, self.position, axis=0)

    def build_trajs_spline(self):
        def remove_close_points(traj, heading, threshold=1.0):
            new_traj = [traj[0]]
            last_point = traj[0]
            for i in range(1, len(traj)):
                if np.linalg.norm(traj[i] - last_point) > threshold:
                    new_traj.append(traj[i])
                    last_point = traj[i]
            if len(new_traj) < 2:
                x_new = traj[0][0] + np.cos(heading)
                y_new = traj[0][1] + np.sin(heading)
                new_traj.append(np.array([x_new, y_new]))
            return np.array(new_traj)
        processed_trajs = remove_close_points(self.planned_trajectory, self.heading)
        self.total_traj_spline = Spline2D(processed_trajs[:, 0], processed_trajs[:, 1])

    @property
    def spline(self):
        return self.total_traj_spline

