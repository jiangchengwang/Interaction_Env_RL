import json
import os
from typing import Optional
from simulation_environment.common.action import Action
from simulation_environment.osm_envs.osm_env import Observation
from simulation_environment.osm_envs.interaction_env import InteractionEnv
from simulation_environment.road.road import Road, RoadNetwork
from simulation_environment.vehicle.humandriving import HumanLikeVehicle
from simulation_environment.road.lane import LineType, StraightLane, PolyLane, PolyLaneFixedWidth
import numpy as np
from utils.state2bev import vehicle_coordinate_sys, absolute_coordinate_sys

import logger
log = logger.get_logger(__name__)


class InteractionGAILEnv(InteractionEnv):
    def __init__(self, data_path=None, osm_path=None, config: dict = None, render_mode: Optional[str] = None):
        super().__init__(data_path, osm_path, config, render_mode)

    @classmethod
    def default_config(self):
        # 设置默认的配置
        config = super().default_config()
        config.update({
            "observation": {"type": "BEV"},
            "screen_width": 640,  # [px]
            "screen_height": 640,  # [px]
            'collision_check': True,
        })
        return config

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        abs_position, abs_velocity, abs_yaw = absolute_coordinate_sys(self.vehicle.position, self.vehicle.speed,
                                                                      self.vehicle.heading,
                                                                      action['rel_position'] if isinstance(action, dict) else action[:2],
                                                                      action['rel_velocity'] if isinstance(action, dict) else action[2:-1],
                                                                      action['rel_yaw'] if isinstance(action, dict) else action[-1])

        self.vehicle.planned_heading[self.vehicle.sim_steps] = abs_yaw
        self.vehicle.planned_speed[self.vehicle.sim_steps] = np.linalg.norm(abs_velocity)
        self.vehicle.planned_trajectory[self.vehicle.sim_steps] = abs_position

        self.road.act()
        self.road.step(1 / self.config["simulation_frequency"])
        self.steps += 1
        self._clear_vehicles()  # 清除车辆
        self.enable_auto_render = False  # 关闭自动渲染
        self._create_bv_vehicles(self.steps)  # 创建虚拟车辆

    def _reward(self, action):
        return 0.0

    def _is_terminated(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        # s, d = self.vehicle.spline.frenet_2_cartesian(
        #     s=self.vehicle.spline.get_s(self.vehicle.position),
        #     d=0)
        # )

        return self.steps >= self.duration or self.vehicle.crashed

    def _is_truncated(self) -> bool:
        """
        Check we truncate the episode at the current step

        :return: is the episode truncated
        """
        return False

    def _clear_vehicles(self) -> None:
        """
        清除车辆
        """
        # 判断车辆是否要离开仿真环境
        is_leaving = lambda vehicle: (self.steps >= (
                    vehicle.planned_trajectory.shape[0] + vehicle.start_step - 1))

        vehicles = []
        for vh in self.road.vehicles:
            try:
                if vh in self.controlled_vehicles or not is_leaving(vh):  # 如果车辆是受控车辆或不需要离开，则保留在环境中
                    vehicles.append(vh)
            except Exception as e:
                print(e)

        self.road.vehicles = vehicles  # 清除需要离开的车辆