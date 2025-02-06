import os
import sys
from data_process.abstract import DataProcess
import pandas as pd
import numpy as np


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.uint8)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        # For other types that are not JSON serializable
        return obj.__str__()


class InterActionDP(DataProcess):
    def __init__(self, config: dict):

        super(InterActionDP, self).__init__(config)

    def read_data(self):
        # 读取csv文件数据
        self.data = pd.read_csv(self.config["dataset_path"])

        # 如果需要处理行人数据
        if self.config['process_pedestrian']:
            ped_data_path = self.config["dataset_path"].replace('vehicle', 'pedestrian')
            if os.path.exists(ped_data_path):
                # 读取行人数据
                ped_data = pd.read_csv(self.config["dataset_path"].replace('vehicle', 'pedestrian')) #TODO　为行人添加朝向与长宽信息
                vx = ped_data['vx'].values
                vy = ped_data['vy'].values
                yaw = np.round(np.arctan2(vy, vx), 3).tolist()
                sup_info = pd.DataFrame({'width': [0.5]*ped_data.shape[0],
                                        'length': [0.5]*ped_data.shape[0],
                                        'psi_rad': yaw })
                ped_data = pd.concat([ped_data, sup_info], axis=1)
                self.data = pd.concat([self.data, ped_data], axis=0)

        self.data[self.config['id_name']] = self.data[self.config['id_name']].astype(str)

def convert_keys(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # 处理键
            if isinstance(k, np.integer):
                k = int(k)
            elif isinstance(k, np.floating):
                k = float(k)
            elif isinstance(k, np.bool_):
                k = bool(k)
            else:
                k = str(k)
            # 递归处理值
            new_dict[k] = convert_keys(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_keys(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj