import os
import re
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)

import json
import time
import logger
from argparse import ArgumentParser
from multiprocessing import cpu_count, Pool

from data_process.data_process import InterActionDP, convert_keys
from data_process.parse_dataset_path import get_all_foler_path

log = logger.setup_app_level_logger()


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="", help="Interaction dataset path", required=True)
    parser.add_argument('--save-path', type=str, default="data/trajectory_set")
    parser.add_argument('--max-cpu-n', type=int, default=32)
    args = parser.parse_args()
    return args


def mutilTDP(file_path, save_path, location_name):
    cf = {
        "dataset_path": file_path,
        "coordinate_x_name": "x",
        "coordinate_y_name": "y",
        "speed_x_name": "vx",
        "speed_y_name": "vy",
        "time_name": "timestamp_ms",
        "frame_id_name": "frame_id",
        "id_name": "track_id",
        "agent_type_name": "agent_type",
        'length_name': 'length',
        'width_name': 'width',
        'heading_name': 'psi_rad',
        "agent_type": ["car"],
        'process_pedestrian': True,  # True则需要处理行人数据
    }
    dataset = InterActionDP(cf)
    dataset()
    log.info("Processing {}.".format(file_path))
    for tp_id in dataset.get_parti_set(dataset.data):  #
        if tp_id.startswith("P"):
            continue
        trajectory_set = dataset.build_trajecotry(0, tp_id)  # 构建轨迹集合
        # 将 trajectory_set 转换为可序列化的形式
        trajectory_set_converted = convert_keys(trajectory_set)
        # 将轨迹集合保存为JSON文件
        save_json_path = os.path.join(project_dir, save_path, location_name)

        with open(os.path.join(save_json_path,
                               f'{file_path.split(os.sep)[-1].split(".")[0]}_trajectory_set_{tp_id}.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(trajectory_set_converted, f, ensure_ascii=False)

    log.info("Successfully processed {} file.".format(file_path))


if __name__ == '__main__':

    start_time = time.time()
    args = make_args()

    if not os.path.exists(f"{project_dir}/{args.save_path}"):
        os.makedirs(f"{project_dir}/{args.save_path}")

    foler_path_info = get_all_foler_path(args.dataset_path)

    n_cpus = min(cpu_count(), args.max_cpu_n)
    log.info(f"CPU allowed: {n_cpus}")

    p = Pool(n_cpus)
    t_start = time.time()  # 记录当前时间
    log.info(f"总共要处理 {len(foler_path_info):d} 个路口数据.")
    for location_name, v in foler_path_info.items():
        save_json_path = os.path.join(project_dir, args.save_path, location_name)
        if not os.path.exists(save_json_path):
            os.makedirs(save_json_path)
        for i, file_path in enumerate(v['data_file']):
            p.apply_async(mutilTDP, args=(file_path, args.save_path, location_name))

    p.close()
    p.join()

    log.info(f"整个脚本的执行时间: {time.time()-start_time:.2f}秒")