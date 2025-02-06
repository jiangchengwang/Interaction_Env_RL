import math
from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
import numpy as np


class Record(ABC):
    """某时刻特征"""

    def __init__(self, **kwargs):
        # 使用 **kwargs 接收不定数量的参数，并将其以键值对的形式保存在对象的实例变量中
        # 根据传入的参数更新对象的属性
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        # 初始化对象的一些属性
        self.id = None  # 记录的唯一 ID
        self.track_id = None  # 轨迹 ID
        self.frame_ID = None  # 帧 ID
        self.unix_time = None  # Unix 时间戳

    def get(self, k):
        # 定义一个 get 方法，用来获取对象的属性
        return self.__getattribute__(k)

    def set(self, k, v):
        # 定义一个 set 方法，用来设置对象的属性
        self.__setattr__(k, v)

    def __deepcopy__(self):
        # 定义一个深拷贝方法，用来创建并返回对象的深度副本
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, v)
        return result


class SnapShot(ABC):
    """某时刻的所有车辆"""

    def __init__(self, **kwargs):
        # 使用 **kwargs 接收不定数量的参数，并将其以键值对的形式保存在对象的实例变量中
        # 根据传入的参数更新对象的属性
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        # 初始化对象的一些属性
        self.unix_time = None  # Unix 时间戳
        self.vr_list = list()  # 保存车辆记录的列表

    def get(self, k):
        # 定义一个 get 方法，用来获取对象的属性
        return self.__getattribute__(k)

    def set(self, k, v):
        # 定义一个 set 方法，用来设置对象的属性
        self.__setattr__(k, v)

    def add_vehicle_record(self, record):
        # 定义一个方法，用来向车辆记录列表中添加记录
        self.vr_list.append(record)

    # @abstractmethod
    # def sort_vehs(self, ascending=True):
    #     raise NotImplementedError


GLB_TIME_THRES = 20000


class TrafficParticipant(ABC):
    """某辆车的所有特征"""

    def __init__(self, **kwargs):
        # 使用 **kwargs 接收不定数量的参数，并将其以键值对的形式保存在对象的实例变量中
        # 根据传入的参数更新对象的属性
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        # 初始化对象的一些属性
        self.id = None  # ID
        self.vr_list = list()  # 保存车辆记录的列表
        self.trajectory = list()  # 保存车辆轨迹的列表
        self.trajectory_ = list()  # 保存划分完成的车辆轨迹的列表

    def get(self, k):
        # 定义一个 get 方法，用来获取对象的属性
        return self.__getattribute__(k)

    def set(self, k, v):
        # 定义一个 set 方法，用来设置对象的属性
        self.__setattr__(k, v)

    def build_trajectory(self, trajectory: list):
        # 定义一个方法，用来构建车辆轨迹
        self.trajectory.extend(trajectory)

    def add_vehicle_record(self, record: Record):
        # 定义一个方法，用来向车辆记录列表中添加记录
        self.vr_list.append(record)

    def build_trajectory_(self):
        # 定义一个方法，用来划分车辆轨迹

        vr_list = self.vr_list
        assert (len(vr_list) > 0)

        self.trajectory_ = list()  # 清空之前的划分结果
        cur_time = vr_list[0].unix_time
        tmp_trj = [vr_list[0]]

        for tmp_vr in vr_list[1:]:
            if tmp_vr.unix_time - cur_time > GLB_TIME_THRES:  # 如果时间戳大于阈值，则将当前轨迹片段添加到轨迹列表中，并开始构建新的轨迹片段
                if len(tmp_trj) > 1:
                    self.trajectory_.append(tmp_trj)
                tmp_trj = [tmp_vr]
            else:
                tmp_trj.append(tmp_vr)
            cur_time = tmp_vr.unix_time

        if len(tmp_trj) > 1:  # 处理最后一个轨迹片段
            self.trajectory_.append(tmp_trj)


class DataProcess(ABC):
    """提取数据集的所有车辆、行人特征"""

    def __init__(self,
                 config: dict = None,
                 snapshot=SnapShot,
                 tparticiant=TrafficParticipant,
                 record=Record,
                 ):

        self.config = self.default_config()
        if config:
            self.config.update(config)

        self.data = None  # 存储读取的数据(pandas.DataFrame)用于交通参与者原始数据

        self.snapshot = snapshot
        self.tparticiant = tparticiant
        self.record = record

        self.vr_dict = dict()  # 信息记录字典
        self.snap_dict = dict()  # 数据快照字典
        self.tp_dict = dict()  # 车辆字典

        self.snap_ordered_list = list()  # 排序后的数据快照列表
        self.tp_ordered_list = list()  # 排序后的交通参与者


    @classmethod
    def default_config(cls):
        '''
        默认配置参数。

        Returns:
        - dict, 默认配置参数字典。
        '''
        return {
            "dataset_path": None,
            "coordinate_x_name": "",
            "coordinate_y_name": "",
            "speed_x_name": "",
            "speed_y_name": "",
            "time_name": "",
            "frame_id_name": "",
            "id_name": "",
            'length_name': '',
            'width_name': '',
            'heading_name': '',
            "agent_type_name": "",
            'process_pedestrian': False
        }

    @abstractmethod
    def read_data(self):
        '''
        读取数据的抽象方法。
        '''
        raise NotImplementedError

    @classmethod
    def separate_data_by_key_value(cls, data, key, value):
        try:
            data_copy = data[data[key] == value].copy()
        except Exception as e:
            raise e
        else:
            return data_copy

    def make_dataframe_become_dict(self, df: pd.DataFrame) -> dict:
        index = self.get_parti_set(df)
        dict_data = dict()
        for idx in index:
            vehicle_data = self.get_the_only_parti(df, idx)
            dict_data[idx] = vehicle_data
        return dict_data

    def get_the_only_parti(self, df: pd.DataFrame, tp_id: str) -> pd.DataFrame:
        return df[df[self.config["id_name"]] == tp_id].copy()

    def get_parti_set(self, df) -> set:
        index = df[[self.config["id_name"]]].values.tolist()
        index = set([idx[0] for idx in index])
        return index

    def process(self):
        '''
        数据处理方法。
        '''

        # 获取数据的列名
        list_name = self.data.columns.tolist()

        # 将DataFrame转换为字典
        tp_dict = self.make_dataframe_become_dict(self.data)

        counter = 0
        self.vr_dict = dict()
        self.snap_dict = dict()
        self.tp_dict = dict()

        # 遍历每个键值对，其中k为字典的键，vehicle为字典的值
        for k, tp in tp_dict.items():
            # 对vehicle进行拷贝并排序
            tp_copy = tp.copy().sort_values(by=self.config['frame_id_name']).reset_index(drop=True)

            # 创建车辆对象
            parti = self.tparticiant()

            # 设置车辆ID
            parti.set('id', tp_copy.loc[0, self.config["id_name"]])
            parti.set('type', tp_copy.loc[0, self.config["agent_type_name"]])

            # 构建车辆轨迹
            parti.build_trajectory(tp_copy.loc[:, [self.config['coordinate_x_name'],
                                                     self.config['coordinate_y_name']]].to_numpy().tolist())

            for i in range(tp_copy.shape[0]):
                vrecord = dict()
                for n in list_name:
                    vrecord[n] = tp_copy.loc[i, n]
                vrecord[self.config['id_name']] = vrecord[self.config['id_name']]
                # 创建记录对象
                tmp_vr = self.record(**vrecord)
                tmp_vr.set('id', counter)
                tmp_vr.set('track_id', tp_copy.loc[i, self.config["id_name"]])
                tmp_vr.set('frame_ID', int(tp_copy.loc[i, self.config["frame_id_name"]]))
                tmp_vr.set('unix_time', int(tp_copy.loc[i, self.config["time_name"]]))

                # 将记录对象添加到vr_dict中
                self.vr_dict[tmp_vr.id] = tmp_vr
                counter += 1

                # 如果unix_time不在snap_dict的键中，则创建一个快照对象，并将其添加到snap_dict中
                if tmp_vr.unix_time not in self.snap_dict.keys():
                    ss = self.snapshot()
                    ss.set('unix_time', tmp_vr.unix_time)
                    self.snap_dict[tmp_vr.unix_time] = ss

                # 将记录对象添加到对应的快照对象和车辆对象中
                self.snap_dict[tmp_vr.unix_time].add_vehicle_record(tmp_vr)
                parti.add_vehicle_record(tmp_vr)
            else:
                self.tp_dict[k] = parti
        else:
            # 将snap_dict的键排序后，存储到snap_ordered_list中
            self.snap_ordered_list = list(self.snap_dict.keys())
            self.snap_ordered_list.sort()

            self.tp_ordered_list = list(self.tp_dict.keys())

    # 可对该方法进行重载
    def __call__(self, *args, **kwargs):

        # 读数据
        self.read_data()
        # 数据处理
        self.process()

    def build_trajecotry(self, period, tp_id):
        assert tp_id in list(self.tp_dict.keys())

        surroundings = []
        record_trajectory = {'ego': {'length': 0, 'width': 0, 'trajectory': [], 'agent_type': ''}}

        for veh_ID, v in self.tp_dict.items():
            v.build_trajectory_()

        ego_trajectories = self.tp_dict[tp_id].trajectory_
        selected_trajectory = ego_trajectories[period]

        D = 200  # 周围车辆的范围

        ego = []  # 存储自车信息的列表
        nearby_IDs = []  # 存储周围车辆id的列表

        for position in selected_trajectory:
            record_trajectory['ego']['length'] = position.get(self.config['length_name'])
            record_trajectory['ego']['width'] = position.get(self.config['width_name'])
            record_trajectory['ego']['agent_type'] = position.get(self.config['agent_type_name'])
            speed = np.linalg.norm(
                [position.get(self.config['speed_x_name']), position.get(self.config['speed_y_name'])])
            ego.append(
                [position.get(self.config['coordinate_x_name']), position.get(self.config['coordinate_y_name']), speed,
                 position.get(self.config['heading_name']),
                 position.get(self.config['time_name'])])  # 将位置、速度、朝向、时间戳添加到ego列表中

            records = self.snap_dict[position.unix_time].vr_list  # 获取指定时间点的记录
            other = []  # 存储其他车辆信息的列表
            for record in records:
                if record.get(self.config['id_name']) != tp_id:
                    other.append(
                        [record.get(self.config['id_name']), record.get(self.config['length_name']),
                         record.get(self.config['width_name']), record.get(self.config['coordinate_x_name']),
                         record.get(self.config['coordinate_y_name']), record.get(self.config['heading_name']),
                         record.get(self.config['speed_x_name']),
                         record.get(self.config['speed_y_name']),
                         record.get(self.config['time_name'])])  # 将其他交通参与者的相关信息添加到other列表中

                    # 构建了包含 x 和 y 坐标差的列表，以计算两点之间的直线距离
                    d = np.linalg.norm([
                        position.get(self.config['coordinate_x_name']) - record.get(self.config['coordinate_x_name']),
                        position.get(self.config['coordinate_y_name']) - record.get(self.config['coordinate_y_name'])
                    ])

                    if d <= D:
                        nearby_IDs.append(record.get(self.config['id_name']))  # 若距离小于等于D，则将id添加到nearby_IDs列表中

            surroundings.append(other)

        record_trajectory['ego']['trajectory'] = ego  # 将ego轨迹添加到record_trajectory字典中

        for v_ID in set(nearby_IDs):
            record_trajectory[v_ID] = {'length': 0, 'width': 0,
                                       'trajectory': [], 'agent_type': ''}

        # 填充数据
        for timestep_record in surroundings:
            scene_IDs = []
            for vehicle_record in timestep_record:
                v_ID = vehicle_record[0]
                v_length = vehicle_record[1]
                v_width = vehicle_record[2]
                v_x = vehicle_record[3]
                v_y = vehicle_record[4]
                v_heading = vehicle_record[5]
                speed = np.linalg.norm([vehicle_record[6], vehicle_record[7]])
                time_stamp = vehicle_record[8]
                if v_ID in set(nearby_IDs):
                    scene_IDs.append(v_ID)
                    record_trajectory[v_ID]['length'] = v_length
                    record_trajectory[v_ID]['width'] = v_width
                    record_trajectory[v_ID]['agent_type'] = self.tp_dict[v_ID].get('type')
                    record_trajectory[v_ID]['trajectory'].append([v_x, v_y, speed,
                                                                  v_heading,
                                                                  time_stamp])

            for v_ID in set(nearby_IDs):
                if v_ID not in scene_IDs:
                    record_trajectory[v_ID]['trajectory'].append([0, 0, 0, 0, 0])

        return record_trajectory