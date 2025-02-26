import json
import os
from typing import Optional
from simulation_environment.common.action import Action
from simulation_environment.osm_envs.osm_env import AbstractEnv, Observation
from simulation_environment.road.road import Road, RoadNetwork
from simulation_environment.vehicle.humandriving import HumanLikeVehicle
from simulation_environment.road.lane import LineType, StraightLane, PolyLane, PolyLaneFixedWidth
import numpy as np
from utils.state2bev import vehicle_coordinate_sys, absolute_coordinate_sys

import logger
log = logger.get_logger(__name__)


class InteractionEnv(AbstractEnv):
    """
    一个带有交互数据的十字路口驾驶环境，用于收集gail训练数据。
    """
    def __init__(self, data_path=None, osm_path=None, config: dict = None, render_mode: Optional[str] = None):

        self.data_path = data_path
        self.file_names = os.listdir(self.data_path)[:]
        # 读取pickle文件中的轨迹数据
        self.trajectory_set = None
        # 获取自车的长度和宽度
        self.ego_length = None
        self.ego_width = None
        # 获取自车的轨迹数据和持续时间
        self.ego_trajectory = None
        self.duration = None
        # 获取周围车辆的ID
        self.surrounding_vehicles = None
        self.vehicle_type = HumanLikeVehicle
        super(InteractionEnv, self).__init__(osm_path=osm_path, config=config, render_mode=render_mode)

    @classmethod
    def default_config(self):
        # 设置默认的配置
        config = super().default_config()
        config.update({
            "observation": {"type": "BEV"},
            "screen_width": 640,  # [px]
            "screen_height": 640,  # [px]
            'collision_check': False,
        })
        return config

    def load_data(self):
        file_name = np.random.choice(self.file_names, size=1, replace=False)[0]  # 随机选择一个文件
        print(f"Load data from {file_name}")
        path = os.path.join(self.data_path, str(file_name))
        # 打开包含路径的文件
        f = open(path, 'rb')
        # 读取pickle文件中的轨迹数据
        self.trajectory_set = json.load(f)
        f.close()
        # 获取自车的长度和宽度
        self.ego_length = self.trajectory_set['ego']['length']
        self.ego_width = self.trajectory_set['ego']['width']
        # 获取自车的轨迹数据和持续时间
        self.ego_trajectory = self.process_raw_trajectory(self.trajectory_set['ego']['trajectory'])
        self.duration = len(self.ego_trajectory) - 1
        # 获取周围车辆的ID
        self.surrounding_vehicles = list(self.trajectory_set.keys())
        self.surrounding_vehicles.pop(0)

    def _reset(self):
        if not self.config['make_road']:
            self._create_road()
            self._create_vehicles()
        else:
            self.config['make_road'] = True

    def _create_road(self):
        # 创建道路
        net = RoadNetwork()
        none = LineType.NONE
        for i, (k, lane) in enumerate(self.roads_dict['lanes'].items()):
            reversed_lane = lane.get('reverse_lane')
            reversed_lane_spline = lane.get('reverse_lane_spline')
            center = lane['center_points']
            left = lane['left']['points']
            right = lane['right']['points']
            net.add_lane(f'{k}_0', f'{k}_1', PolyLane(center, left, right, line_types=(none, none), lane_id=k, reverse_lane_id=reversed_lane, reverse_lane_spline=reversed_lane_spline))

        # 加载人行道
        pedestrian_marking_id = 0
        for k, v in self.roads_dict['way'].items():
            ls = v['type']
            if ls["type"] == "pedestrian_marking":
                pedestrian_marking_id += 1
                ls_points = v['points']
                net.add_lane(f"P_{pedestrian_marking_id}_start", f"P_{pedestrian_marking_id}_end",
                             PolyLaneFixedWidth(ls_points, line_types=(none, none), width=5))

        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"], lanelet=self.roads_dict, collision_checker=self.config['collision_check'])

    def process_raw_trajectory(self, trajectory):
        """
        处理原始轨迹，将坐标、速度进行转换。
        :param trajectory: 原始轨迹数据
        :return: 转换后的轨迹数据
        """
        trajectory = np.array(trajectory).copy()
        shape = trajectory.shape
        trajectory = trajectory.reshape(-1, shape[-1])
        x, y = trajectory[:, 0].copy(), trajectory[:, 1].copy()
        trajectory[:, 0] = y
        trajectory[:, 1] = x
        headings = trajectory[:, 3].copy()
        headings = np.pi / 2 - headings
        headings = (headings + np.pi) % (2 * np.pi) - np.pi
        trajectory[:, 3] = headings
        return trajectory.reshape(shape)

    def _create_vehicles(self, reset_time=0):
        """
        创建自车和NGSIM车辆，并将它们添加到道路上。
        """
        self.load_data()

        T = 100  # 设置T的值
        self.controlled_vehicles = []  # 初始化受控车辆列表
        whole_trajectory = self.ego_trajectory  # 获取整个轨迹
        ego_trajectory = np.array(whole_trajectory[reset_time:])  # 获取自车轨迹
        self.vehicle = self.vehicle_type(self.road, 'ego', ego_trajectory[0][:2], ego_trajectory[0][3], ego_trajectory[0][2],
                                        ngsim_traj=ego_trajectory, target_velocity=ego_trajectory[1][2],
                                        v_length=self.trajectory_set['ego']['length'], v_width=self.trajectory_set['ego']['width'])  # 创建自车实例
        # target = self.road.network.get_closest_lane_index(position=ego_trajectory[-1][:2])  # 获取目标车道
        # self.vehicle.plan_route_to(target[1])  # 规划车辆行驶路线
        self.vehicle.color = (50, 200, 0)  # 设置车辆颜色
        self.road.vehicles.append(self.vehicle)  # 将车辆添加到道路上
        self._create_bv_vehicles(self.steps)  # 创建虚拟车辆

    def _create_bv_vehicles(self, current_time):

        reset_time = 0
        T = 200

        vehicles = []  # 初始化其他车辆列表
        for veh_id in self.surrounding_vehicles:  # 遍历周围车辆
            try:
                other_trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])  # 获取其他车辆轨迹
                flag = ~(np.array(other_trajectory[current_time])).reshape(1, -1).any(axis=1)[0]  # 判断是否存在轨迹点
                if current_time == 0:  # 如果当前时间为0
                    pass
                else:
                    trajectory = np.array(self.trajectory_set[veh_id]['trajectory'][reset_time:])  # 获取当前时间轨迹点
                    if not flag and ~(np.array(trajectory[current_time-1])).reshape(1, -1).any(axis=1)[0]:  # 如果当前时间和上一时间存在轨迹点
                        flag = False
                    else:
                        flag = True

                if not flag:  # 如果不存在轨迹点

                    mask = np.where(np.sum(other_trajectory[:(T * 10), :2], axis=1) == 0, False, True)  # 过滤无效轨迹点
                    other_trajectory = self.process_raw_trajectory(other_trajectory[mask])
                    if other_trajectory.shape[0] <= 5:  # 如果计划轨迹点不足5个，则跳过
                        continue
                    other_vehicle = self.vehicle_type(self.road, f"{veh_id}", other_trajectory[0][:2], other_trajectory[0][3], other_trajectory[0][2],
                                     ngsim_traj=other_trajectory, target_velocity=other_trajectory[1][2], start_step=self.steps,
                                     v_length=self.trajectory_set[veh_id]['length'],
                                     v_width=self.trajectory_set[veh_id]['width'])

                    vehicles.append(other_vehicle)  # 将其他车辆添加到列表中

            except Exception as e:  # 捕获异常
                raise ValueError(f'Error in creating other vehicles, error {e}')
        else:
            if len(vehicles) > 0:
                for vh in self.road.vehicles:
                    vehicles.append(vh)  # 将其他车辆添加到列表中
                self.road.vehicles = vehicles  # 将道路上的车辆替换为新的车辆列表

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
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
        return self.steps >= self.duration

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


class InteractionV1Env(InteractionEnv):

    def __init__(self, data_path=None, osm_path=None, config: dict = None, render_mode: Optional[str] = None,
                 bv_ids=None):
        self.data_path = data_path
        with open(data_path, "r") as f:
            self.trajectory_set = json.load(f)
        # 获取自车的长度和宽度
        self.ego_length = self.trajectory_set['ego']['length']
        self.ego_width = self.trajectory_set['ego']['width']
        # 获取自车的轨迹数据和持续时间
        self.ego_trajectory = self.process_raw_trajectory(self.trajectory_set['ego']['trajectory'])
        self.duration = len(self.ego_trajectory) - 1
        # 获取周围车辆的ID
        self.surrounding_vehicles = list(self.trajectory_set.keys())
        self.surrounding_vehicles.pop(0)
        self.bv_ids = bv_ids
        self.vehicle_type = HumanLikeVehicle
        super(InteractionEnv, self).__init__(osm_path=osm_path, config=config, render_mode=render_mode)

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        收集环境信息
        :param obs: 观测
        :param action: 动作
        :return: 环境信息
        """
        info = super(InteractionEnv, self)._info(obs, action)
        destination_position = self.vehicle.planned_trajectory[self.steps+1]
        destination_speed = self.vehicle.planned_speed[self.steps+1]
        destination_heading = self.vehicle.planned_heading[self.steps+1]

        rel_position, rel_velocity, rel_yaw = vehicle_coordinate_sys(self.vehicle.position, self.vehicle.speed,
                                                             self.vehicle.heading,
                                                             destination_position, destination_speed,
                                                             destination_heading)

        abs_position, abs_velocity, abs_yaw = absolute_coordinate_sys(self.vehicle.position, self.vehicle.speed,
                                                                      self.vehicle.heading,
                                                                      rel_position, rel_velocity,
                                                                      rel_yaw)
        log.info(f'steps: {self.steps}' )
        log.info(f'distance: {np.linalg.norm(abs_position - destination_position)}' )
        log.info(f'speed: {np.linalg.norm(np.linalg.norm(abs_velocity) - destination_speed)}')
        log.info(f'yaw: {abs_yaw - destination_heading}', )

        info['action'] = {
            'rel_position': rel_position.tolist(),
            'rel_velocity': rel_velocity.tolist(),
            'rel_yaw': float(rel_yaw),
            'action': self.vehicle.action,
        }
        log.info(f"action: {info['action']}")
        log.info('------------------')

        return info

    def _create_vehicles(self, reset_time=0):
        """
        创建自车和NGSIM车辆，并将它们添加到道路上。
        """
        self.controlled_vehicles = []  # 初始化受控车辆列表
        whole_trajectory = self.ego_trajectory  # 获取整个轨迹
        ego_trajectory = np.array(whole_trajectory[reset_time:])  # 获取自车轨迹
        self.vehicle = self.vehicle_type(self.road, 'ego', ego_trajectory[0][:2], ego_trajectory[0][3], ego_trajectory[0][2],
                                        ngsim_traj=ego_trajectory, target_velocity=ego_trajectory[1][2],
                                        v_length=self.trajectory_set['ego']['length'], v_width=self.trajectory_set['ego']['width'])  # 创建自车实例
        self.vehicle.color = (50, 200, 0)  # 设置车辆颜色
        self.road.vehicles.append(self.vehicle)  # 将车辆添加到道路上

        self._create_bv_vehicles(self.steps)  # 创建虚拟车辆

    def _is_terminated(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        return self.steps >= self.duration - 1


class InteractionEvalEnv(InteractionV1Env):

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        abs_position, abs_velocity, abs_yaw = absolute_coordinate_sys(self.vehicle.position, self.vehicle.speed,
                                                                      self.vehicle.heading,
                                                                      action['rel_position'],
                                                                      action['rel_velocity'],
                                                                      action['rel_yaw'])

        self.vehicle.planned_heading[self.vehicle.sim_steps] = abs_yaw
        self.vehicle.planned_speed[self.vehicle.sim_steps] = np.linalg.norm(abs_velocity)
        self.vehicle.planned_trajectory[self.vehicle.sim_steps] = abs_position
        print('prediction: ')
        print('distance: ', np.linalg.norm(abs_position - self.vehicle.planned_trajectory[self.vehicle.sim_steps]))
        print('speed: ', np.linalg.norm(np.linalg.norm(abs_velocity) - self.vehicle.planned_speed[self.vehicle.sim_steps]))
        print('yaw: ', abs_yaw - self.vehicle.planned_heading[self.vehicle.sim_steps])
        print('------------------')

        self.road.act()
        self.road.step(1 / self.config["simulation_frequency"])
        self.steps += 1
        self._clear_vehicles()  # 清除车辆
        self.enable_auto_render = False  # 关闭自动渲染
        self._create_bv_vehicles(self.steps)  # 创建虚拟车辆