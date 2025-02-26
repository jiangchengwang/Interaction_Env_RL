import os
import sys
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)
from data_process.parse_dataset_path import get_all_foler_path, get_all_trajectory_set_data
from utils.font_path import chinese_font_path, english_font_path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import simulation_environment
import gymnasium as gym
from diffusion_policy.bev_dataset import BEVDataset
from diffusion_policy.networks import *
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

import cv2

english_font = font_manager.FontProperties(fname=english_font_path, size=16)
chinese_font = font_manager.FontProperties(fname=chinese_font_path, size=16)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, default='', help="Interaction dataset path", required=True)
    parser.add_argument('--train-dataset', type=str, default='data/dataset')
    parser.add_argument('--trajectory-path', type=str, default="data/trajectory_set")
    parser.add_argument('--model-path', type=str, default='data/model', help="Experiment data path")
    args = parser.parse_args()

    dataset_file_info = get_all_foler_path(args.dataset_path)
    trajecotry_file_info = get_all_trajectory_set_data(os.path.join(project_dir, args.trajectory_path))

    location_name = 'DR_CHN_Roundabout_LN'
    v = trajecotry_file_info[location_name]
    osm_path = dataset_file_info[location_name]['map_file']
    eval_dataset = BEVDataset(dataset_path=os.path.join(project_dir, args.train_dataset), train=False)

    #load model
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    obs_dim = vision_feature_dim
    action_dim = (1, 5)

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim[0],
        global_cond_dim=obs_dim
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # device transfer
    device = torch.device('cuda')

    nets.load_state_dict(torch.load(os.path.join(project_dir, args.model_path, 'best_model.pt')))

    _ = nets.to(device)

    for data_path in trajecotry_file_info[location_name]['trajectory_file_path']:
        env = gym.make(
            'interaction-eval-v0',
            render_mode="rgb_array",
            data_path=data_path,
            osm_path=osm_path,
        )
        # obs rgb
        obs, info = env.reset()
        done = False
        while not done:
            nimages = eval_dataset.normalize_obs(obs)
            nimages = torch.from_numpy(nimages).to(device, dtype=torch.float32).unsqueeze(0)
            # infer action
            with torch.no_grad():
                # get image features
                image_features = nets['vision_encoder'](nimages)
                obs_cond = image_features

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (1, 5), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = nets['noise_pred_net'](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
            naction = naction.detach().to('cpu').numpy().squeeze(0)
            naction = eval_dataset.unnormalize_data(naction)

            action = {
                'rel_position': naction[:2],
                'rel_velocity': naction[ 2:-1],
                'rel_yaw': naction[-1],
            }
            env.render()
            cv2.imshow('image', obs)
            cv2.waitKey(1)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        env.close()
