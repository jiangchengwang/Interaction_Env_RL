from enum import IntEnum, Enum
import networkx as nx
import pickle


class WayPoint(object):
    def __init__(self):
        # 经度
        self.lon: float = -1.0
        # 维度
        self.lat: float = -1.0
        # 东北天 x y z
        # x坐标
        self.x: float = -1.0
        # y坐标
        self.y: float = -1.0
        # 高度 m
        self.z: float = -1.0
        # heading
        self.heading: float = -1.0
        # width
        self.width = -1.0
        # width
        self.length = -1.0
        # 车道上的点
        self.lane_id = None
        # road id
        self.road_id = None
        # section id
        self.section_id = None
        # junction id
        self.junction_id = None
        # id
        self.id = None


class RoadOption(IntEnum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.

    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6

class GRouterPlanner(object):

    def __init__(self, sampling_resolution=3, decimal=1):
        self.graph = None
        self.id_map = None
        self.road_id_to_edge = None
        self.topology = None

        self.kdtree = None
        self.id_wpt = dict()

        self.decimal = decimal
        self._sampling_resolution = sampling_resolution

        self._build_topology()
        self._build_kdtree()
        self._build_graph()

    def _build_topology(self):
        raise NotImplementedError

    def _build_kdtree(self):
        raise NotImplementedError

    def _build_graph(self):
        raise NotImplementedError

    def _localize(self, point):
        raise NotImplementedError

    def _distance_heuristic(self, n1, n2):
        raise NotImplementedError

    def _path_search(self, origin, destination):
        start, end = self._localize(origin), self._localize(destination)

        route = nx.astar_path(self.graph, source=start[0], target=end[0], heuristic=self._distance_heuristic, weight='length')
        route.append(end[1])
        return route

    def path_search(self, origin, destination):
        routes = [origin[0]]
        route = []
        try:
            route = nx.astar_path(self.graph, source=origin[1], target=destination[0], heuristic=self._distance_heuristic,
                                  weight='length')
        except Exception as e:
            print(e)
        routes.extend(route)
        routes.append(destination[1])
        return routes

    def pickle(self):
        # pass
        f = open(f'map.pickle', 'wb')
        pickle.dump(self, f)
        f.close()