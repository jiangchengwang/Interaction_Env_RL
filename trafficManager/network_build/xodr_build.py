# for: NetworkBuild with Frenet
from __future__ import annotations

import random
import sys
from typing import Any, Union, List, Tuple

import numpy as np
from lxml import etree
from scipy.spatial import KDTree
from trafficManager.network_build.opendrive_parse.parser import parse_opendrive
from trafficManager.network_build.xodr_road_graph import Road, SectionLane, NormalLane
from trafficManager.network_build.opendrive_parse.network import LinkIndex
import logger
log = logger.get_logger(__name__)


def encode_road_section_lane_width_id(roadId, sectionId, laneId, widthId):
    """

    Args:
      roadId:
      sectionId:
      laneId:
      widthId:

    Returns:

    """
    return ".".join([str(roadId), str(sectionId), str(laneId), str(widthId)])


def decode_road_section_lane_width_id(encodedString: str):
    """

    Args:
      encodedString:

    Returns:

    """

    parts = encodedString.split(".")

    if len(parts) != 4:
        raise Exception()

    return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))


class geoHash:
    def __init__(self, id: tuple[int]) -> None:
        self.id = id
        self.edges: set[str] = set()


class NetworkBuild(object):
    def __init__(self,
                 networkFile: str = None,
                 obsFile: str = None
                 ) -> None:
        self.networkFile = networkFile
        self.obsFile = obsFile
        self.openDriveXml = None
        self.roads: dict[str, 'Road'] = {}
        self.lane_sections: dict[str, 'SectionLane'] = {}
        self.lanes: dict[str, 'NormalLane'] = {}
        self.geoHashes: dict[tuple[int], geoHash] = {}
        self._planes = []
        self._link_index = None

    def affGridIDs(self, centerLine: list[tuple[float]]) -> set[tuple[int]]:
        affGridIDs = set()
        for poi in centerLine:
            poixhash = int(poi[0] // 10)
            poiyhash = int(poi[1] // 10)
            affGridIDs.add((poixhash, poiyhash))

        return affGridIDs

    def load_opendrive(self, xml_text: str = None):
        """Load all elements of an OpenDRIVE network to a parametric lane representation

        Args:
            opendrive:

        """
        if self.networkFile is not None:
            fh = open(self.networkFile, "r")
            self.openDriveXml = parse_opendrive(etree.parse(fh).getroot())
            fh.close()
        elif xml_text is not None:
            self.openDriveXml = parse_opendrive(etree.fromstring(xml_text.encode('utf-8')))
        else:
            raise ValueError(f'No OpenDrive file or xml text provided')

        self._link_index = LinkIndex()
        self._link_index.create_from_opendrive(self.openDriveXml)

    def calc_precision(self, length: float) -> float:
        precision = 0
        if length < 5:
            precision = 0.1
        elif length < 20:
            precision = 0.5
        # elif length < 100:
        #     precision = 1.0
        # elif length < 500:
        #     precision = 2.0
        # elif length < 1000:
        #     precision = 3.0
        else:
            precision = 1.0

        return precision

    def getData(self):
        # Convert all parts of a road to parametric lanes (planes)

        for road in self.openDriveXml.roads:

            precision = self.calc_precision(road.planView.length)
            road.planView.precalculate(precision)
            if road.planView.length < 1.0:
                continue
            laneoffsets = []
            laneoffset_len = len(road.lanes.laneOffsets)
            laneoffset_idx = 0
            for i, (ds, x, y, yaw) in enumerate(road.planView.get_precalculation):
                if laneoffset_len == 0:
                    laneoffset = 0
                else:
                    if road.lanes.laneOffsets[laneoffset_idx].start_pos < ds:
                        while laneoffset_idx < laneoffset_len-1:
                            if road.lanes.laneOffsets[laneoffset_idx + 1].start_pos > ds:
                                break
                            laneoffset_idx += 1

                    laneoffset = sum(coeff * (ds - road.lanes.laneOffsets[laneoffset_idx].start_pos) ** i for i, coeff in
                                     enumerate(road.lanes.laneOffsets[laneoffset_idx].polynomial_coefficients))

                laneoffsets.append(laneoffset)

            start_pre_precalculation_idx = 0
            start_long_off = 0.0
            for lanesection in road.lanes.lane_sections:
                lanesection.process(road.planView.get_precalculation, start_pre_precalculation_idx, precision,
                                    start_long_off, laneoffsets)
                start_pre_precalculation_idx = lanesection.end_pre_precalculation_idx
                start_long_off += lanesection.length

    def build_graph(self):
        for rd in self.openDriveXml.roads:
            road = Road(rd.id)
            section_lanes = {}
            for lsection in rd.lanes.lane_sections:
                lane_section = SectionLane(lsection.idx, rd.id)
                lanes = {}
                for ln in lsection.allLanes:
                    if ln.id == 0:
                        continue
                    lane = NormalLane(ln.id, lsection.idx, rd.id)
                    # start_idx = lsection.lane_start_end_label[ln.id]['start_idx']
                    # end_idx = lsection.lane_start_end_label[ln.id]['end_idx']
                    center_position = lsection.lane_center_dict[ln.id]
                    lane_width = lsection.lane_width_dict[ln.id]
                    lane.in_junction = True if rd.junction is not None else False
                    lane.getPlotElem(center_position, lane_width)
                    lane.lane_type = ln.type
                    lanes[ln.id] = lane
                    self.lanes[f'{rd.id}.{lsection.idx}.{ln.id}'] = lane
                    laneAffGridIDs = self.affGridIDs(center_position)
                    road.affGridIDs = road.affGridIDs | laneAffGridIDs

                lane_section.lanes = lanes
                section_lanes[lsection.idx] = lane_section
                self.lane_sections[f'{rd.id}.{lsection.idx}'] = lane_section

            road.lane_sections = section_lanes
            for gridID in road.affGridIDs:
                try:
                    geohash = self.geoHashes[gridID]
                except KeyError:
                    geohash = geoHash(gridID)
                    self.geoHashes[gridID] = geohash
                geohash.edges.add(rd.id)
            self.roads[rd.id] = road

        for connecting, successors in self._link_index._successors.copy().items():
            try:
                from_road_id, from_lane_section_idx, from_lane_link_successorId, _ = decode_road_section_lane_width_id(connecting)
            except Exception:
                self._link_index._successors.pop(connecting)
                # log.error(f'Decoding connecting {connecting} failed.')

            new_cuccessors = []
            for successor in successors:

                try:
                    road_id, lane_section_idx, lane_link_successorId, _ = decode_road_section_lane_width_id(successor)
                    new_cuccessors.append(successor)

                except Exception:
                    # log.error(f'Decoding successor {successor} failed.')
                    continue

            self._link_index._successors[connecting] = new_cuccessors

        for road in self.openDriveXml.roads:
            for lanesection in road.lanes.lane_sections:
                for i, lane in enumerate(lanesection.allLanes):
                    if lane.type in ['driving'] and lane.id != 0:
                        if road.junction is not None:
                            continue

                        left_lane = lane.get_left_lane()
                        if left_lane is not None and left_lane.type in ['driving']:
                            self.lanes[f'{road.id}.{lanesection.idx}.{lane.id}'].left_lane_id = left_lane.id

                        right_lane = lane.get_right_lane()
                        if right_lane is not None and right_lane.type in ['driving']:
                            self.lanes[f'{road.id}.{lanesection.idx}.{lane.id}'].right_lane_id = right_lane.id

    def get_next_lanes(self, road_id: str, section_lane_id: str, lane_id: str) -> List[Tuple[str, str, str]]:
        successors = self._link_index._successors.get(f'{road_id}.{section_lane_id}.{lane_id}.-1')
        res = []
        if successors is not None:
            for successor in successors:
                road_id, section_lane_id, lane_id, _ = decode_road_section_lane_width_id(successor)
                res.append((road_id, section_lane_id, lane_id))

        return res

    def get_available_next_lane(self, road_info: Tuple, available_lanes: list[Tuple]):
        next_lanes = []
        for next_lane_i in self.get_next_lanes(*road_info):

            if next_lane_i in available_lanes:
                next_lane = self.get_lane_by_id(*next_lane_i)
                next_lanes.append(next_lane)
                if next_lane.length <= 30:
                    for next_lane_j in self.get_next_lanes(*next_lane_i):
                        if next_lane_j in available_lanes:
                            next_lanes.append(self.get_lane_by_id(*next_lane_j))
                return next_lanes

            next_lane_i_left = self.get_left_lane_id(*next_lane_i)
            next_lane_i_right = self.get_right_lane_id(*next_lane_i)
            if next_lane_i_left in available_lanes or next_lane_i_right in available_lanes:
                next_lanes.append(self.get_lane_by_id(*next_lane_i))
                return next_lanes

        return [self.get_lane_by_id(*road_info)]

    def get_road(self, rid: str):
        return self.roads.get(rid)

    def get_left_lane_id(self, road_id: str, section_lane_id: str, lane_id: str) -> Tuple:

        lane = self.lanes.get(f'{road_id}.{section_lane_id}.{lane_id}')
        left_lane_id = lane.left_lane_id if lane.left_lane_id is not None else lane_id
        return road_id, section_lane_id, left_lane_id

    def get_right_lane_id(self, road_id: str, section_lane_id: str, lane_id: str) -> Tuple:

        lane = self.lanes.get(f'{road_id}.{section_lane_id}.{lane_id}')
        right_lane_id = lane.right_lane_id if lane.right_lane_id is not None else lane_id
        return road_id, section_lane_id, right_lane_id

    def get_all_lanes(self, road_id: str, section_lane_id: str, lane_id: str) -> List[NormalLane]:
        section_lanes = self.lane_sections.get(f'{road_id}.{section_lane_id}')
        section_lanes = [v.to_tuple() for k, v in section_lanes.lanes.items() if np.sign(k) == np.sign(int(lane_id))]
        return section_lanes

    def get_lane_by_id(self, road_id: str, section_lane_id: str, lane_id: str) -> NormalLane:
        return self.lanes.get(f'{road_id}.{section_lane_id}.{lane_id}')

    def process(self):
        if self.networkFile is not None:
            self.load_opendrive()
        self.getData()
        self.build_graph()
        log.info('Network build finished')
