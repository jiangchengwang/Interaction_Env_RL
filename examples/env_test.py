import os
import sys
import time
import logger
from argparse import ArgumentParser
from data_process.parse_dataset_path import get_all_foler_path, get_all_trajectory_set_data
import simulation_environment
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
import cv2

log = logger.setup_app_level_logger()


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default="", help="Interaction dataset path", required=True)
    parser.add_argument('--trajectory-path', type=str, default="data/trajectory_set")
    parser.add_argument('--save-video', action='store_true', help="Save video")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = make_args()

    dataset_file_info = get_all_foler_path(args.dataset_path)
    trajecotry_file_info = get_all_trajectory_set_data(os.path.join(project_dir, args.trajectory_path))

    for location_name, v in trajecotry_file_info.items():

        data_path = str(np.random.choice(trajecotry_file_info[location_name]['trajectory_file_path']))
        osm_path = dataset_file_info[location_name]['map_file']
        env = gym.make(
            'interaction-v1',
            render_mode="rgb_array",
            data_path=data_path,
            osm_path=osm_path,
        )
        if args.save_video:
            env = RecordVideo(
                env, video_folder="./videos", episode_trigger=lambda e: True,
                name_prefix=location_name,
            )
            env.unwrapped.set_record_video_wrapper(env)

        obs, info = env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(None)
            env.render()
            done = terminated or truncated
            # bgr_img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            cv2.imshow('Test', obs)
            cv2.waitKey(1)
        env.close()