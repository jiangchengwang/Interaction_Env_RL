from __future__ import annotations

from collections import OrderedDict, deque

from itertools import product
from typing import TYPE_CHECKING, List, Any, Optional, Callable, Dict


import numpy as np
import pandas as pd

from gymnasium import spaces
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from simulation_environment import utils
from simulation_environment.road.lane import AbstractLane
from simulation_environment.vehicle.kinematics import Vehicle
from utils.state2bev import polygon_xy_from_state, draw_vehicle, draw_pedestrian, vehicle_coordinate_sys

matplotlib.use('agg')

try:
    import torch
    from torch_geometric.data import Dataset
    from torch_geometric.data import HeteroData
    from torch_geometric.data import extract_tar
except ImportError:
    print('torch_geometric is not installed, please install it by running "pip install torch-geometric"')

if TYPE_CHECKING:
    from simulation_environment.common.abstract import AbstractEnv


class ObservationType:
    def __init__(self, env: AbstractEnv, **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class KinematicObservation(ObservationType):
    """Observe the kinematics of nearby vehicles."""

    FEATURES: list[str] = ["presence", "x", "y", "vx", "vy"]

    def __init__(
        self,
        env: AbstractEnv,
        features: list[str] = None,
        vehicles_count: int = 5,
        features_range: dict[str, list[float]] = None,
        absolute: bool = False,
        order: str = "sorted",
        normalize: bool = True,
        clip: bool = True,
        see_behind: bool = False,
        observe_intentions: bool = False,
        include_obstacles: bool = True,
        **kwargs: dict,
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        self.include_obstacles = include_obstacles

    def space(self) -> spaces.Space:
        return spaces.Box(
            shape=(self.vehicles_count, len(self.features)),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
        )

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(
                self.observer_vehicle.lane_index
            )
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [
                    -AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                    AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                ],
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])
        # Add nearby traffic
        close_vehicles = self.env.road.close_objects_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
            sort=self.order == "sorted",
            vehicles_only=not self.include_obstacles,
        )
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            vehicles_df = pd.DataFrame.from_records(
                [
                    v.to_dict(origin, observe_intentions=self.observe_intentions)
                    for v in close_vehicles[-self.vehicles_count + 1 :]
                ]
            )
            df = pd.concat([df, vehicles_df], ignore_index=True)

        df = df[self.features]

        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            )
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs.astype(self.space().dtype)


class EmptyObservation(ObservationType):
    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy', 'heading']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 6,
                 features_range: dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = False,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 include_obstacles: bool = True,
                 shape: tuple[int, int] = None,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        self.include_obstacles = include_obstacles
        self.shape = shape

    # def space(self) -> spaces.Space:
    #     return spaces.Box(shape=(1, 56), low=-np.inf, high=np.inf, dtype=np.float32)

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.shape, low=-np.inf, high=np.inf,
                          dtype=np.float32)

    def observe(self, veh: Vehicle = None) -> Any:
        if not self.env.road:
            return np.zeros(self.space().shape, np.float32)

        return self.env.get_obs()


class QCNetInteractionDataset(ObservationType):
    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy', 'heading']

    def __init__(self, env: 'AbstractEnv',
                 features: List[str] = None,
                 vehicles_count: int = 6,
                 features_range: dict[str, List[float]] = None,
                 absolute: bool = False,
                 order: str = "sorted",
                 normalize: bool = False,
                 clip: bool = True,
                 see_behind: bool = True,
                 observe_intentions: bool = False,
                 include_obstacles: bool = True,
                 shape: tuple[int, int] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 3,
                 num_historical_steps: int = 50,
                 num_future_steps: int = 60,
                 predict_unseen_agents: bool = False,
                 vector_repr: bool = True,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        self.include_obstacles = include_obstacles
        self.shape = shape

        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.predict_unseen_agents = predict_unseen_agents
        self.vector_repr = vector_repr
        self.transform = transform

        self._agent_types = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
                             'construction', 'riderless_bicycle', 'unknown']
        self._agent_categories = ['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']
        self._polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
        self._polygon_is_intersections = [True, False, None]
        self._point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                             'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                             'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
                             'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
        self._point_sides = ['LEFT', 'RIGHT', 'CENTER']
        self._polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']
        self.vehicle_history_traj : Dict[deque] = {}

    # def space(self) -> spaces.Space:
    #     return spaces.Box(shape=(1, 56), low=-np.inf, high=np.inf, dtype=np.float32)

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.shape, low=-np.inf, high=np.inf,
                          dtype=np.float32)

    def updata_vehicle_trajectory(self, vehicle_name):
        timestamp = self.env.steps
        curernt_vehicle_names = set()
        for veh in self.env.road.vehicles:
            x = veh.position[0]
            y = veh.position[1]
            heading = veh.heading
            vx = veh.velocity[0]
            vy = veh.velocity[1]
            object_type = 'vehicle'
            track_id = veh.get_name
            curernt_vehicle_names.add(track_id)
            if track_id not in self.vehicle_history_traj.keys():
                self.vehicle_history_traj[track_id] = deque(maxlen=self.num_historical_steps)

            if track_id == 'ego' or track_id == vehicle_name:
                object_category = 3
            else:
                object_category = 2

            self.vehicle_history_traj[track_id].append([x, y, vx, vy, heading, object_type, track_id, timestamp, object_category])

        # remove the vehicle that has not been seen for 50 steps
        remove_vehicle_name = self.vehicle_history_traj.keys() - curernt_vehicle_names
        for name in remove_vehicle_name:
            top = self.vehicle_history_traj[name][-1]
            if self.env.steps - top[-2] >= 50:
                self.vehicle_history_traj.pop(name)

    def observe(self, vehicle_name: str = None) -> Any:
        if not self.env.road:
            return np.zeros(self.space().shape, np.float32)

        if vehicle_name is None:
            vehicle_name = self.env.vehicle.name
        self.updata_vehicle_trajectory(vehicle_name)
        data = dict()
        data['city'] = 'Interaction'
        data['agent'] = self.get_agent_features(vehicle_name)
        data.update(self.get_map_features())
        return HeteroData(data)

    def get_agent_features(self, vehicle_name):
        df = pd.DataFrame.from_records(list(self.vehicle_history_traj[vehicle_name]),
                                       columns=['position_x', 'position_y', 'velocity_x', 'velocity_y', 'heading',
                                                'object_type', 'track_id', 'timestep', 'object_category'])

        for track_id in self.vehicle_history_traj.keys():
            if track_id == vehicle_name:
                continue

            df = pd.concat([df, pd.DataFrame.from_records(list(self.vehicle_history_traj[track_id]),
                                                          columns=['position_x', 'position_y', 'velocity_x', 'velocity_y', 'heading',
                                                                   'object_type', 'track_id', 'timestep', 'object_category'])],
                           ignore_index=True)
        if self.env.steps > 49:
            df['timestep'] = df['timestep'] - (self.env.steps - 49)
        historical_df = df[df['timestep'] < self.num_historical_steps]
        agent_ids = list(historical_df['track_id'].unique())
        df = df[df['track_id'].isin(agent_ids)]
        num_agents = len(agent_ids)
        av_idx = agent_ids.index(vehicle_name)

        # initialization
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_id: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        for track_id, track_df in df.groupby('track_id'):
            agent_idx = agent_ids.index(track_id)
            agent_steps = track_df['timestep'].values

            valid_mask[agent_idx, agent_steps] = True
            current_valid_mask[agent_idx] = valid_mask[agent_idx, self.num_historical_steps - 1]
            predict_mask[agent_idx, agent_steps] = True
            if self.vector_repr:  # a time step t is valid only when both t and t-1 are valid
                valid_mask[agent_idx, 1: self.num_historical_steps] = (
                        valid_mask[agent_idx, :self.num_historical_steps - 1] &
                        valid_mask[agent_idx, 1: self.num_historical_steps])
                valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, :self.num_historical_steps] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps:] = False

            agent_id[agent_idx] = track_id
            agent_type[agent_idx] = self._agent_types.index(track_df['object_type'].values[0])
            agent_category[agent_idx] = track_df['object_category'].values[0]
            position[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['position_x'].values,
                                                                              track_df['position_y'].values],
                                                                             axis=-1)).float()
            heading[agent_idx, agent_steps] = torch.from_numpy(track_df['heading'].values).float()
            velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['velocity_x'].values,
                                                                              track_df['velocity_y'].values],
                                                                             axis=-1)).float()

        predict_mask[current_valid_mask | (agent_category == 2) | (agent_category == 3), self.num_historical_steps:] = True

        return {
            'num_nodes': num_agents,
            'av_index': av_idx,
            'valid_mask': valid_mask,
            'predict_mask': predict_mask,
            'id': agent_id,
            'type': agent_type,
            'category': agent_category,
            'position': position,
            'heading': heading,
            'velocity': velocity,
        }

    def get_mark_type(self, line_type):

        if line_type['type'] == 'virtual':
            mark_type = 'NONE'
        elif line_type['type'] == 'curbstone':
            mark_type = 'SOLID_YELLOW'
        else:
            subtype = line_type['subtype']
            color = line_type['color']
            if subtype == 'solid_solid':
                subtype = 'double_solid'
            elif subtype == 'dash_dash':
                subtype = 'double_dash'

            mark_type = subtype.upper() + '_' + color.upper()

        return mark_type

    def get_map_features(self):
        lane_segment_ids = list(self.env.roads_dict['lanes'].keys())
        cross_walk_ids = []
        polygon_ids = lane_segment_ids + cross_walk_ids
        num_polygons = len(lane_segment_ids) + len(cross_walk_ids) * 2

        # initialization
        polygon_position = torch.zeros(num_polygons, self.dim, dtype=torch.float)
        polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
        polygon_height = torch.zeros(num_polygons, dtype=torch.float)
        polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
        polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
        point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
        point_side: List[Optional[torch.Tensor]] = [None] * num_polygons

        for k, lane_segment in self.env.roads_dict['lanes'].items():
            lane_segment_idx = polygon_ids.index(k)
            centerline = torch.from_numpy(np.array(lane_segment['center_points'])).float()
            polygon_position[lane_segment_idx] = centerline[0, :self.dim]
            polygon_orientation[lane_segment_idx] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                                centerline[1, 0] - centerline[0, 0])
            polygon_height[lane_segment_idx] = centerline[1, 2] - centerline[0, 2]
            polygon_type[lane_segment_idx] = self._polygon_types.index('VEHICLE')
            polygon_is_intersection[lane_segment_idx] = self._polygon_is_intersections.index(
                lane_segment['is_intersection'])

            left_boundary = torch.from_numpy(np.array(lane_segment['left']['points'])).float()
            right_boundary = torch.from_numpy(np.array(lane_segment['right']['points'])).float()
            point_position[lane_segment_idx] = torch.cat([left_boundary[:-1, :self.dim],
                                                          right_boundary[:-1, :self.dim],
                                                          centerline[:-1, :self.dim]], dim=0)
            left_vectors = left_boundary[1:] - left_boundary[:-1]
            right_vectors = right_boundary[1:] - right_boundary[:-1]
            center_vectors = centerline[1:] - centerline[:-1]
            point_orientation[lane_segment_idx] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                             torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                             torch.atan2(center_vectors[:, 1], center_vectors[:, 0])],
                                                            dim=0)
            point_magnitude[lane_segment_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                      right_vectors[:, :2],
                                                                      center_vectors[:, :2]], dim=0), p=2, dim=-1)
            point_height[lane_segment_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                       dim=0)

            left_type = self._point_types.index(self.get_mark_type(lane_segment['left']['type']))
            right_type = self._point_types.index(self.get_mark_type(lane_segment['right']['type']))
            center_type = self._point_types.index('CENTERLINE')
            point_type[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), left_type, dtype=torch.uint8),
                 torch.full((len(right_vectors),), right_type, dtype=torch.uint8),
                 torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
            point_side[lane_segment_idx] = torch.cat(
                [torch.full((len(left_vectors),), self._point_sides.index('LEFT'), dtype=torch.uint8),
                 torch.full((len(right_vectors),), self._point_sides.index('RIGHT'), dtype=torch.uint8),
                 torch.full((len(center_vectors),), self._point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

        num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
        point_to_polygon_edge_index = torch.stack(
            [torch.arange(num_points.sum(), dtype=torch.long),
             torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
        polygon_to_polygon_edge_index = []
        polygon_to_polygon_type = []

        def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
            try:
                return ls.index(elem)
            except ValueError:
                return None

        for k, lane_segment in self.env.roads_dict['lanes'].items():
            lane_segment_idx = polygon_ids.index(k)
            pred_inds = []
            for pred in list(lane_segment['predecessors']):
                pred_idx = safe_list_index(polygon_ids, pred)
                if pred_idx is not None:
                    pred_inds.append(pred_idx)
            if len(pred_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                                 torch.full((len(pred_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(pred_inds),), self._polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
            succ_inds = []
            for succ in list(lane_segment['successors']):
                succ_idx = safe_list_index(polygon_ids, succ)
                if succ_idx is not None:
                    succ_inds.append(succ_idx)
            if len(succ_inds) != 0:
                polygon_to_polygon_edge_index.append(
                    torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                                 torch.full((len(succ_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
                polygon_to_polygon_type.append(
                    torch.full((len(succ_inds),), self._polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))
            if lane_segment['left_lane_id'] is not None:
                left_idx = safe_list_index(polygon_ids, lane_segment['left_lane_id'])
                if left_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('LEFT')], dtype=torch.uint8))
            if lane_segment['right_lane_id'] is not None:
                right_idx = safe_list_index(polygon_ids, lane_segment['right_lane_id'])
                if right_idx is not None:
                    polygon_to_polygon_edge_index.append(
                        torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long))
                    polygon_to_polygon_type.append(
                        torch.tensor([self._polygon_to_polygon_types.index('RIGHT')], dtype=torch.uint8))
        if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
        map_data['map_polygon']['num_nodes'] = num_polygons
        map_data['map_polygon']['position'] = polygon_position
        map_data['map_polygon']['orientation'] = polygon_orientation
        if self.dim == 3:
            map_data['map_polygon']['height'] = polygon_height
        map_data['map_polygon']['type'] = polygon_type
        map_data['map_polygon']['is_intersection'] = polygon_is_intersection
        if len(num_points) == 0:
            map_data['map_point']['num_nodes'] = 0
            map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
            map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
            map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
        else:
            map_data['map_point']['num_nodes'] = num_points.sum().item()
            map_data['map_point']['position'] = torch.cat(point_position, dim=0)
            map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
            map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
            if self.dim == 3:
                map_data['map_point']['height'] = torch.cat(point_height, dim=0)
            map_data['map_point']['type'] = torch.cat(point_type, dim=0)
            map_data['map_point']['side'] = torch.cat(point_side, dim=0)
        map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
        map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

        return map_data


class BEVObservation(ObservationType):
    FEATURES: list[str] = ["presence", "x", "y", "vx", "vy", "heading"]

    def __init__(
            self,
            env: AbstractEnv,
            features: list[str] = None,
            vehicles_count: int = 5,
            features_range: dict[str, list[float]] = None,
            absolute: bool = False,
            order: str = "sorted",
            normalize: bool = True,
            clip: bool = True,
            see_behind: bool = False,
            observe_intentions: bool = False,
            include_obstacles: bool = True,
            **kwargs: dict,
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        self.include_obstacles = include_obstacles

    def space(self) -> spaces.Space:
        return spaces.Box(
            shape=(480, 480, 3),
            low=0,
            high=255,
            dtype=np.uint8,
        )

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        def batch_coordinates_conversion(base_position, base_heading, coordinates):
            base_position = np.array(base_position).reshape(-1, 2)
            delta_p = coordinates - base_position

            # 转换坐标系：旋转-A的朝向角
            rot_matrix = np.array([
                [np.cos(-base_heading), -np.sin(-base_heading)],
                [np.sin(-base_heading), np.cos(-base_heading)]
            ])
            rel_position = rot_matrix @ delta_p.T

            return rel_position.T

        fig, ax = plt.subplots(figsize=(4.8, 2.4))
        ax.set_aspect('equal')
        ax.set_xlim(-20, 80)
        ax.set_ylim(-25, 25)
        # ax.set_facecolor('gray')
        ego_position = self.env.vehicle.position
        ego_heading = self.env.vehicle.heading
        ego_speed = self.env.vehicle.speed

        ego_destination = self.env.vehicle.planned_trajectory[-1].copy()
        rel_destination = batch_coordinates_conversion(ego_position, ego_heading, ego_destination.reshape(-1, 2))
        type_dict = dict(color="red", linewidth=3, zorder=10)
        rel_destination = np.stack([np.zeros((1, 2)), rel_destination], axis=1).squeeze(axis=0)
        plt.plot(rel_destination[:, 0], rel_destination[:, 1], **type_dict)

        # draw traffic participant
        for veh in self.env.road.vehicles:
            veh_position = veh.position
            veh_heading = veh.heading
            veh_speed = veh.speed
            rel_position, rel_velocity, rel_yaw = vehicle_coordinate_sys(ego_position, ego_speed, ego_heading, veh_position, veh_speed, veh_heading)
            color = 'blue' if veh.name != 'ego' else 'green'
            draw_vehicle(ax, rel_position[0], rel_position[1], rel_yaw, veh.LENGTH, veh.WIDTH, color=color)

        # draw map
        for ls in self.env.laneletmap.lineStringLayer:

            if "type" not in ls.attributes.keys():
                raise RuntimeError("ID " + str(ls.id) + ": Linestring type must be specified")
            elif ls.attributes["type"] == "curbstone":
                type_dict = dict(color="black", linewidth=1, zorder=10)
            elif ls.attributes["type"] == "line_thin":
                if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                    type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[10, 10])
                else:
                    type_dict = dict(color="white", linewidth=1, zorder=10)
            elif ls.attributes["type"] == "line_thick":
                if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                    type_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
                else:
                    type_dict = dict(color="white", linewidth=2, zorder=10)
            elif ls.attributes["type"] == "pedestrian_marking":
                type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
            elif ls.attributes["type"] == "bike_marking":
                type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
            elif ls.attributes["type"] == "stop_line":
                type_dict = dict(color="white", linewidth=3, zorder=10)
            elif ls.attributes["type"] == "virtual":
                type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[2, 5])
            elif ls.attributes["type"] == "road_border":
                type_dict = dict(color="black", linewidth=1, zorder=10)
            elif ls.attributes["type"] == "guard_rail":
                type_dict = dict(color="black", linewidth=1, zorder=10)
            elif ls.attributes["type"] == "traffic_sign":
                continue
            elif ls.attributes["type"] == "building":
                type_dict = dict(color="pink", zorder=1, linewidth=5)
            elif ls.attributes["type"] == "spawnline":
                if ls.attributes["spawn_type"] == "start":
                    type_dict = dict(color="green", zorder=11, linewidth=2)
                elif ls.attributes["spawn_type"] == "end":
                    type_dict = dict(color="red", zorder=11, linewidth=2)

            ls_points = [(pt.y, pt.x) for pt in ls]
            ls_points = np.array(ls_points)
            ls_points = batch_coordinates_conversion(ego_position, ego_heading, ls_points)
            plt.plot(ls_points[:, 0], ls_points[:, 1], **type_dict)

        lanelets = []
        for ll in self.env.laneletmap.laneletLayer:
            points = [[pt.y, pt.x] for pt in ll.polygon2d()]
            points = np.array(points)
            points = batch_coordinates_conversion(ego_position, ego_heading, points)
            polygon = Polygon(points.tolist(), True)
            lanelets.append(polygon)

        ll_patches = PatchCollection(lanelets, facecolors="lightgray", edgecolors="None", zorder=5)
        ax.add_collection(ll_patches)

        if len(self.env.laneletmap.laneletLayer) == 0:
            ax.patch.set_facecolor('lightgrey')

        areas = []
        for area in self.env.laneletmap.areaLayer:
            if area.attributes["subtype"] == "keepout":
                points = [[pt.y, pt.x] for pt in area.outerBoundPolygon()]
                points = np.array(points)
                points = batch_coordinates_conversion(ego_position, ego_heading, points)
                polygon = Polygon(points.tolist(), True)
                areas.append(polygon)

        area_patches = PatchCollection(areas, facecolors="darkgray", edgecolors="None", zorder=5)
        ax.add_collection(area_patches)

        # 关闭坐标轴和边框
        ax.axis('off')
        # 获取当前图像数据
        fig.canvas.draw()
        # 从图形对象中提取图像数据
        rgba_buf = fig.canvas.buffer_rgba()
        (w, h) = fig.canvas.get_width_height()
        rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
        rgba_arr = rgba_arr[:, :, :-1]
        return rgba_arr


def observation_factory(env: AbstractEnv, config: dict) -> ObservationType:
    if config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "Empty":
        return EmptyObservation(env, **config)
    elif config["type"] == "QCNet":
        return QCNetInteractionDataset(env, **config)
    elif config["type"] == "BEV":
        return BEVObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")