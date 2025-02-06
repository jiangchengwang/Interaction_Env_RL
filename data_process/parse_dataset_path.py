import os
import os.path as osp
import subprocess


def get_all_foler_path(path: str):
    if not osp.isdir(path):
        raise ValueError("path is not a folder")

    folder_name = subprocess.getoutput(f"find {path} -name 'vehicle_tracks*csv'").split('\n')
    folder_name = sorted(folder_name)
    res = {}
    for i, name in enumerate(folder_name):
        location_name = name.split('/')[-2]
        if res.get(location_name) is None:
            res[location_name] = {}
            res[location_name]['data_file'] = []
            res[location_name]['map_file'] = os.path.join(os.sep.join(name.split('/')[:-3]), 'maps', location_name + '.osm')
            if not osp.isfile(res[location_name]['map_file']):
                raise ValueError("map file is not exist")
        res[location_name]['data_file'].append(name)

    return res


def get_all_trajectory_set_data(path: str):
    if not osp.isdir(path):
        raise ValueError("path is not a folder")

    folder_name = subprocess.getoutput(f"find {path} -name '*json'").split('\n')
    folder_name = sorted(folder_name)
    res = {}
    for i, name in enumerate(folder_name):
        location_name = name.split('/')[-2]
        if res.get(location_name) is None:
            res[location_name] = {}
            res[location_name]['trajectory_file_path'] = []
        res[location_name]['trajectory_file_path'].append(name)

    return res