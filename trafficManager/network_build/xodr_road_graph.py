
from __future__ import annotations
from utils.cubic_spline import Spline2D
import numpy as np
from typing import Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC
from scipy.spatial import KDTree
import logger

log = logger.get_logger(__name__)
OVERLAP_DISTANCE = 0.1  # junction overlap distance

@dataclass
class Junction:
    id: str = None
    incoming_roads: set[str] = field(default_factory=set)
    outgoing_roads: set[str] = field(default_factory=set)
    affGridIDs: set[tuple[int]] = field(default_factory=set)
    shape: list[tuple[float]] = None

@dataclass
class Road:
    id: str = None
    junction_id: str = None
    section_lanes: Dict[str] = field(default_factory=dict)
    from_road: str = None
    to_road: str = None
    next_edge_info: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )  # next edge and the corresponding **self** normal lane
    obstacles: dict = field(default_factory=dict)
    affGridIDs: set[tuple[int]] = field(default_factory=set)
    _waypoints_x: list[float] = None
    _waypoints_y: list[float] = None

    @property
    def section_num(self):
        return len(self.section_lanes)

    def __hash__(self):
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Edge(id={self.id})"
        # return f"Edge(id={self.id}, lane_num={len(self.lanes)}, from_junction={self.from_junction}, to_junction={self.to_junction})\n"


@dataclass
class SectionLane:
    id: str = None
    road_id: str = None
    lanes: Dict[str] = field(default_factory=dict)

    def __hash__(self):
        return hash(f'{self.road_id}_{self.id}')

    def __repr__(self) -> str:
        return f"SectionLane(id={self.id})"
        # return f"Edge(id={self.id}, lane_num={len(self.lanes)}, from_junction={self.from_junction}, to_junction={self.to_junction})\n"

def normalize_angles(angles):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def angle_difference(angle1, angle2):
    return ((angle2 - angle1 + np.pi) % (2 * np.pi)) - np.pi


def calc_cumsum_angle_change(headings):
    headings = normalize_angles(headings)
    # 计算方向角差值并规范化
    delta_thetas = angle_difference(headings[:-1], headings[1:])
    delta_thetas = normalize_angles(delta_thetas)

    # 累计方向角变化并规范化
    cumulative_angles = np.cumsum(delta_thetas)
    cumulative_angles = normalize_angles(cumulative_angles)
    return cumulative_angles


def check_turn_sign(headings, low_threshold=60, up_threshold=120):
    cumulative_angles = calc_cumsum_angle_change(headings)
    # 设置容差
    tolerance = 0.1  # 弧度
    # 检测掉头事件
    u_turns = np.abs(cumulative_angles) >= (np.pi - tolerance)
    u_turn_indices = np.where(u_turns)[0] + 1  # +1调整索引

    # 设置阈值范围 (60° 到 120°) 转弯角度
    low_threshold = np.radians(low_threshold)  # 60度
    up_threshold = np.radians(up_threshold)  # 120度

    # 判断是否发生左转或右转
    left_turns = (cumulative_angles >= low_threshold) & (
            cumulative_angles <= up_threshold)
    right_turns = (cumulative_angles <= -
    low_threshold) & (cumulative_angles >= -up_threshold)
    turn_sign = 'round' if u_turn_indices.shape[0] > 0 else 'left' if np.sum(left_turns) > 0 else 'right' if np.sum(
        right_turns) > 0 else 'straight'
    # turn_sign = 'straight'
    # if u_turn_indices.shape[0] > 0:
    #     turn_sign = 'round'
    # elif np.sum(left_turns) > 0:
    #     turn_sign = 'left'
    # elif np.sum(right_turns) > 0:
    #     turn_sign = 'right'

    return turn_sign


@dataclass
class AbstractLane(ABC):
    """
    Abstract lane class.
    """
    id: str
    section_id: str
    road_id: str
    speed_limit: float = 13.89  # m/s
    length: float = 0
    course_spline: Spline2D = None
    left_lane_id: str = None
    right_lane_id: str = None
    lane_type: list = None
    in_junction : bool = False
    turn_sign: str = 'straight'

    @property
    def spline_length(self):
        return self.course_spline.s[-1]

    def getPlotElem(self, center_position, width_list):
        center_position = np.array(center_position)
        self.center_line = []
        self.left_bound = []
        self.right_bound = []
        if int(self.id) > 0:
            center_position = center_position[::-1]
            width_list = width_list[::-1]

        self.course_spline = Spline2D(center_position[:, 0], center_position[:, 1])
        self.length = self.course_spline.s[-1]
        headings = []
        for i in range(center_position.shape[0]):
            w = width_list[i]
            si, _ = self.course_spline.cartesian_to_frenet1D(float(center_position[i, 0]), float(center_position[i, 1]))
            self.center_line.append(self.course_spline.calc_position(float(si)))
            self.left_bound.append(self.course_spline.frenet_to_cartesian1D(float(si), w / 2))
            self.right_bound.append(self.course_spline.frenet_to_cartesian1D(float(si), -w / 2))
            headings.append(self.course_spline.calc_yaw(float(si)))
        if self.in_junction:
            self.turn_sign = check_turn_sign(np.array(headings))
        self.width_kdtree = KDTree(self.left_bound)

    def get_lane_width(self, x, y):
        _, idx = self.width_kdtree.query([x, y])
        nn_data = self.width_kdtree.data[idx]
        lon, lat = self.course_spline.cartesian_to_frenet1D(nn_data[0], nn_data[1])
        return lat * 2

@dataclass
class NormalLane(AbstractLane):
    """
    Normal lane from edge
    """
    affiliated_edge: Road = None
    next_lanes: Dict[str, tuple[str, str]] = field(
        default_factory=dict
    )  # next_lanes[to_lane_id: normal lane] = (via_lane_id, direction)

    def left_lane(self) -> str:
        return None

    def right_lane(self) -> str:
        return None

    def __hash__(self):
        return hash(f'{self.road_id}_{self.section_id}_{self.id}')

    def to_tuple(self):
        return tuple((self.road_id, self.section_id, self.id))

    def __repr__(self) -> str:
        # return f"NormalLane(id={self.id}, width = {self.width})"
        return f"NormalLane(id={self.id})"


@dataclass
class RoadGraph:
    """
    Road graph of the map
    """

    roads: Dict[str, Road] = field(default_factory=dict)

    def get_lane_by_id(self, road_id: str, section_lane_id: str, lane_id: str) -> AbstractLane:
        pass

    def get_next_lane(self, road_id: str, section_lane_id: str, lane_id: str) -> AbstractLane:
        pass

    def get_available_next_lane(self, road_id: str, section_lane_id: str, lane_id: str, available_lanes: list[str]) -> AbstractLane:
        pass

    def __str__(self):
        return 'edges: {}, \nlanes: {}, \njunctions lanes: {}'.format(
            self.edges.keys(), self.lanes.keys(),
            self.junction_lanes.keys()
        )
