import os
import sys
import time
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)

from argparse import ArgumentParser
from data_process.parse_dataset_path import get_all_foler_path, get_all_trajectory_set_data
import simulation_environment
import gymnasium as gym
import pickle
from multiprocessing import cpu_count, Pool

import logger
log = logger.setup_app_level_logger()


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="", help="Interaction dataset path", required=True)
    parser.add_argument('--trajectory-path', type=str, default="data/trajectory_set")
    parser.add_argument('--max-cpu-n', type=int, default=2, help="Number of process")
    args = parser.parse_args()
    return args


def mutilTDP(osm_path, data_path, location_name, tag='observation'):
    nat_dataset = {
        "obs": [],
        "action": [],
    }
    file_name = data_path.split(os.sep)[-1].split(".")[0]
    env = gym.make(
        'interaction-v1',
        render_mode="rgb_array",
        data_path=data_path,
        osm_path=osm_path,
    )
    # obs rgb
    obs, info = env.reset()
    done = False
    while not done:
        nat_dataset["obs"].append(obs)
        nat_dataset["action"].append(info["action"])
        obs, reward, terminated, truncated, info = env.step(None)
        done = terminated or truncated
    env.close()
    with open(os.path.join(project_dir, 'data', tag, location_name, f'{file_name}.pkl'), 'wb') as f:
        pickle.dump(nat_dataset, f)
        log.info(f"Process {file_name} at the {location_name} done.")


if __name__ == '__main__':

    args = make_args()

    dataset_file_info = get_all_foler_path(args.dataset_path)
    trajecotry_file_info = get_all_trajectory_set_data(os.path.join(project_dir, args.trajectory_path))

    n_cpus = min(cpu_count(), args.max_cpu_n)
    log.info(f"CPU allowed: {n_cpus}")

    p = Pool(n_cpus)
    t_start = time.time()  # 记录当前时间
    log.info(f"总共有 {len(dataset_file_info):d} 个路口数据.")
    tag = 'observation'
    # for location_name, v in trajecotry_file_info.items():
    location_name = 'DR_CHN_Roundabout_LN'
    v = trajecotry_file_info[location_name]
    save_path = os.path.join(project_dir, 'data', tag, location_name)
    osm_path = dataset_file_info[location_name]['map_file']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for data_path in trajecotry_file_info[location_name]['trajectory_file_path']:
        p.apply_async(mutilTDP, args=(osm_path, data_path, location_name, tag))

    p.close()
    p.join()

    log.info(f"整个脚本的执行时间: {time.time() - t_start:.2f}秒")





