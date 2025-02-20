# -*- coding: utf-8 -*-
import numpy as np

from trafficManager.network_build.opendrive_parse.elements.road_record import RoadRecord
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import sys
from utils.cubic_spline import Spline2D


class Lanes:
    """ """

    def __init__(self):
        self._laneOffsets = []
        self._lane_sections = []

    @property
    def laneOffsets(self):
        """ """
        self._laneOffsets.sort(key=lambda x: x.start_pos)
        return self._laneOffsets

    @property
    def lane_sections(self):
        """ """
        self._lane_sections.sort(key=lambda x: x.sPos)
        return self._lane_sections

    def getLaneSection(self, laneSectionIdx):
        """

        Args:
          laneSectionIdx:

        Returns:

        """
        for laneSection in self.lane_sections:
            if laneSection.idx == laneSectionIdx:
                return laneSection

        return None

    def getLastLaneSectionIdx(self):
        """ """

        numLaneSections = len(self.lane_sections)

        if numLaneSections > 1:
            return numLaneSections - 1

        return 0


class LaneOffset(RoadRecord):
    """The lane offset record defines a lateral shift of the lane reference line
    (which is usually identical to the road reference line).

    (Section 5.3.7.1 of OpenDRIVE 1.4)

    """


class LeftLanes:
    """ """

    sort_direction = False

    def __init__(self):
        self._lanes = []

    @property
    def lanes(self):
        """ """
        self._lanes.sort(key=lambda x: x.id, reverse=self.sort_direction)
        return self._lanes


class CenterLanes(LeftLanes):
    """ """


class RightLanes(LeftLanes):
    """ """

    sort_direction = True


class Lane:
    """ """

    laneTypes = [
        "none",
        "driving",
        "stop",
        "shoulder",
        "biking",
        "sidewalk",
        "border",
        "restricted",
        "parking",
        "bidirectional",
        "median",
        "special1",
        "special2",
        "special3",
        "roadWorks",
        "tram",
        "rail",
        "entry",
        "exit",
        "offRamp",
        "onRamp",
    ]

    def __init__(self, parentRoad, lane_section):
        self._parent_road = parentRoad
        self._id = None
        self._type = None
        self._level = None
        self._link = LaneLink()
        self._widths = []
        self._borders = []
        self.lane_section = lane_section
        self.has_border_record = False
        self.course_spline = None
        self.lane_width_spline = None

    def getPlotElem(self):
        s = np.linspace(0, self.course_spline.s[-1], num=50)
        for si in s:

            width = self.lane_width_spline.calc_width(si)

            self.center_line = [
                self.course_spline.calc_position(si)
            ]
            self.left_bound = [
                self.course_spline.frenet_to_cartesian1D(si, width / 2)
            ]
            self.right_bound = [
                self.course_spline.frenet_to_cartesian1D(si, -width / 2)
            ]

    def get_left_lane(self):
        if abs(self.id) <= 1:
            return None
        left_lane_id = abs(self.id) - 1
        return self.lane_section.getLane(np.sign(self.id) * left_lane_id)

    def get_right_lane(self):
        if ((self.id > 0 and abs(self.id) >= len(self.lane_section.leftLanes)) \
            or (self.id < 0 and abs(self.id) >= len(self.lane_section.rightLanes))
                or self.id == 0):
            return None
        right_lane_id = abs(self.id) + 1
        return self.lane_section.getLane(np.sign(self.id) * right_lane_id)


    @property
    def parentRoad(self):
        """ """
        return self._parent_road

    @property
    def id(self):
        """ """
        return self._id

    @id.setter
    def id(self, value):
        self._id = int(value)

    @property
    def type(self):
        """ """
        return self._type

    @type.setter
    def type(self, value):
        if value not in self.laneTypes:
            raise Exception()

        self._type = str(value)

    @property
    def level(self):
        """ """
        return self._level

    @level.setter
    def level(self, value):
        if value not in ["true", "false"] and value is not None:
            raise AttributeError("Value must be true or false.")

        self._level = value == "true"

    @property
    def link(self):
        """ """
        return self._link

    @property
    def widths(self):
        """ """
        self._widths.sort(key=lambda x: x.start_offset)
        return self._widths

    @widths.setter
    def widths(self, value):
        """"""
        self._widths = value

    def getWidth(self, widthIdx):
        """

        Args:
          widthIdx:

        Returns:

        """
        for width in self._widths:
            if width.idx == widthIdx:
                return width

        return None

    def getLastLaneWidthIdx(self):
        """Returns the index of the last width sector of the lane"""

        numWidths = len(self._widths)

        if numWidths > 1:
            return numWidths - 1

        return 0

    @property
    def borders(self):
        """ """
        return self._borders


class LaneLink:
    """ """

    def __init__(self):
        self._predecessor = None
        self._successor = None

    @property
    def predecessorId(self):
        """ """
        return self._predecessor

    @predecessorId.setter
    def predecessorId(self, value):
        self._predecessor = int(value)

    @property
    def successorId(self):
        """ """
        return self._successor

    @successorId.setter
    def successorId(self, value):
        self._successor = int(value)


class LaneSection:
    """The lane section record defines the characteristics of a road cross-section.

    (Section 5.3.7.2 of OpenDRIVE 1.4)

    """

    def __init__(self, road=None):
        self.idx = None
        self.sPos = None
        self._singleSide = None
        self._leftLanes = LeftLanes()
        self._centerLanes = CenterLanes()
        self._rightLanes = RightLanes()

        self._parentRoad = road

        self.lane_accu_width_dict = {}
        self.lane_start_end_label = {}
        self.lane_center_dict = {}
        self.start_pre_precalculation_idx = None
        self.end_pre_precalculation_idx = None
        self.lanes_course_spline = None
        self.width_course_spline = None


    @property
    def singleSide(self):
        """Indicator if lane section entry is valid for one side only."""
        return self._singleSide

    @singleSide.setter
    def singleSide(self, value):
        if value not in ["true", "false"] and value is not None:
            raise AttributeError("Value must be true or false.")

        self._singleSide = value == "true"

    @property
    def leftLanes(self):
        """Get list of sorted lanes always starting in the middle (lane id -1)"""
        return self._leftLanes.lanes

    @property
    def centerLanes(self):
        """ """
        return self._centerLanes.lanes

    @property
    def rightLanes(self):
        """Get list of sorted lanes always starting in the middle (lane id 1)"""
        return self._rightLanes.lanes

    @property
    def allLanes(self):
        """Attention! lanes are not sorted by id"""
        return self._leftLanes.lanes + self._centerLanes.lanes + self._rightLanes.lanes

    def getLane(self, lane_id: int) -> Lane:
        """

        Args:
          lane_id:

        Returns:

        """
        for lane in self.allLanes:
            if lane.id == lane_id:
                return lane

        return None

    @property
    def parentRoad(self):
        """ """
        return self._parentRoad

    @property
    def get_lane_center(self, index):
        return self.lane_center_dict[index]

    @property
    def get_lane_start_end_idx(self, index):
        return self.lane_start_end_label[index]

    @property
    def get_lane_width(self, lane_idx, point_idx):
        return self.lane_accu_width_dict[lane_idx][point_idx]

    def process(self, pre_precalculation, start_pre_precalculation_idx, precision, start_long_off, laneoffsets):
        self.lane_accu_width_dict = {}
        self.lane_width_dict = {}
        self.lane_start_end_label = {}
        self.lane_center_dict = {}
        self.start_pre_precalculation_idx = start_pre_precalculation_idx
        self.end_pre_precalculation_idx = int(start_pre_precalculation_idx + (self.length // precision)) + 1
        self.end_pre_precalculation_idx = self.end_pre_precalculation_idx if self.end_pre_precalculation_idx < pre_precalculation.shape[0] \
            else pre_precalculation.shape[0]
        for lane in self.allLanes:
            self.lane_accu_width_dict[lane.id] = []
            self.lane_start_end_label[lane.id] = {}
            if lane.id == 0:
                self.lane_start_end_label[0]['start_idx'] = start_pre_precalculation_idx
                self.lane_start_end_label[0]['end_idx'] = self.end_pre_precalculation_idx
                self.lane_accu_width_dict[0] = [0.0 for i in range(start_pre_precalculation_idx, self.end_pre_precalculation_idx+1)]
                continue

            widths = lane.widths
            counter = 0  # record idx
            for idx in range(self.start_pre_precalculation_idx, self.end_pre_precalculation_idx):
                try:
                    ds, x, y = pre_precalculation[idx, 0], pre_precalculation[idx, 1], pre_precalculation[idx, 2],
                except Exception as e:
                    print(e)
                lane_index = 0
                while lane_index < len(widths):
                    if widths[lane_index].start_offset <= (ds - start_long_off) <= (
                            widths[lane_index].start_offset + widths[lane_index].length):
                        # Evaluate the polynomial equation to get the lane width
                        lane_width = sum(coeff * (ds - widths[lane_index].start_offset - start_long_off) ** i for i, coeff in
                                         enumerate(widths[lane_index].polynomial_coefficients))

                        self.lane_accu_width_dict[lane.id].append(lane_width)
                        counter += 1
                        break
                    lane_index += 1
                if counter == 1:
                    self.lane_start_end_label[lane.id]['start_idx'] = idx

            if counter > 1:
                self.lane_start_end_label[lane.id]['end_idx'] = self.lane_start_end_label[lane.id]['start_idx'] + counter - 1

            if counter <= 1:
                raise ValueError('Counter must be greater than one.')

        left_lane_num = len(self.leftLanes)
        right_lane_num = len(self.rightLanes)

        self.lane_width_dict = deepcopy(self.lane_accu_width_dict)

        for i in range(1, left_lane_num + 1):

            lane_center_line = []

            for j in range(len(self.lane_accu_width_dict[i])):
                idx = self.start_pre_precalculation_idx+j
                ds, x, y, angle = (pre_precalculation[idx, 0], pre_precalculation[idx, 1],
                            pre_precalculation[idx, 2], pre_precalculation[idx, 3])
                # 对齐lane　width　idx
                try:
                    if (self.lane_start_end_label[i]['start_idx']) >= (self.lane_start_end_label[i-1]['start_idx'] + j):
                        self.lane_accu_width_dict[i][j] += self.lane_accu_width_dict[i-1][0]
                    elif (self.lane_start_end_label[i]['start_idx']+j) >= self.lane_start_end_label[i-1]['end_idx']:
                        self.lane_accu_width_dict[i][j] += self.lane_accu_width_dict[i-1][len(self.lane_accu_width_dict[i-1])-1]
                    else:
                        self.lane_accu_width_dict[i][j] += self.lane_accu_width_dict[i-1][
                            (j+self.lane_start_end_label[i]['start_idx']-self.lane_start_end_label[i-1]['start_idx'])]
                except Exception as e:
                    print(e)

                # Calculate the offset from the reference line for the lane center
                offset = self.lane_accu_width_dict[i][j] - self.lane_width_dict[i][j] / 2 + laneoffsets[idx]

                # Calculate the position of the lane center
                center_x = x - offset * math.sin(angle)
                center_y = y + offset * math.cos(angle)
                lane_center_line.append([center_x, center_y])
                idx += 1
            self.lane_center_dict[i] = lane_center_line
            center_line = np.array(lane_center_line.copy())
            if self.length > 1.0:
                self.lanes_course_spline = Spline2D(center_line[:, 0], center_line[:, 1])

        for i in range(-1, -(right_lane_num + 1), -1):
            lane_center_line = []
            try:
                idx = self.lane_start_end_label[i]['start_idx']
            except:
                print('func:{}(),line:{},'.format(sys._getframe().f_code.co_name, sys._getframe().f_lineno), end="")
            j = 0
            while idx <= self.lane_start_end_label[i]['end_idx']:
                ds, x, y, angle = pre_precalculation[idx, 0], pre_precalculation[idx, 1], pre_precalculation[idx, 2],pre_precalculation[idx, 3]
                # 对齐lane　width　idx
                if (self.lane_start_end_label[i]['start_idx']) >= (self.lane_start_end_label[i+1]['start_idx'] + j):
                    self.lane_accu_width_dict[i][j] += self.lane_accu_width_dict[i + 1][0]
                elif (self.lane_start_end_label[i]['start_idx']+j) >= self.lane_start_end_label[i+1]['end_idx']:
                    self.lane_accu_width_dict[i][j] += self.lane_accu_width_dict[i+1][len(self.lane_accu_width_dict[i+1])-1]
                else:
                    self.lane_accu_width_dict[i][j] += self.lane_accu_width_dict[i+1][
                        (j+self.lane_start_end_label[i]['start_idx']-self.lane_start_end_label[i+1]['start_idx'])]

                # Calculate the offset from the reference line for the lane center
                offset = self.lane_accu_width_dict[i][j] - self.lane_width_dict[i][j] / 2 - laneoffsets[idx]

                # Calculate the position of the lane center
                center_x = x + offset * math.sin(angle)
                center_y = y - offset * math.cos(angle)
                lane_center_line.append([center_x, center_y])
                idx += 1
                j += 1
            self.lane_center_dict[i] = lane_center_line



class LaneWidth(RoadRecord):
    """Entry for a lane describing the width for a given position.
    (Section 5.3.7.2.1.1.2 of OpenDRIVE 1.4)


    start_offset being the offset of the entry relative to the preceding lane section record

"""

    def __init__(
        self,
        *polynomial_coefficients: float,
        idx: int = None,
        start_offset: float = None
    ):
        self.idx = idx
        self.length = 0
        super().__init__(*polynomial_coefficients, start_pos=start_offset)

    @property
    def start_offset(self):
        """Return start_offset, which is the offset of the entry to the
        start of the lane section.
        """
        return self.start_pos

    @start_offset.setter
    def start_offset(self, value):
        self.start_pos = value


class LaneBorder(LaneWidth):
    """Describe lane by width in respect to reference path.

    (Section 5.3.7.2.1.1.3 of OpenDRIVE 1.4)

    Instead of describing lanes by their width entries and, thus,
    invariably depending on influences of inner
    lanes on outer lanes, it might be more convenient to just describe
    the outer border of each lane
    independent of any inner lanes’ parameters.
    """
