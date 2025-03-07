import copy
import glob
import os
import sys
import time
from collections import deque

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
import simulation_environment
import gymnasium as gym
import numpy as np
import torch
import json
import argparse
from rl import algo, utils
from rl.algo import gail
from rl.algo.ppo import to_device
from rl.networks import CNNPolicy, CNNDiscriminator, CNNValue
from rl.storage import Memory
from data_process.parse_dataset_path import get_all_foler_path, get_all_trajectory_set_data
import logger
log = logger.setup_app_level_logger(level='INFO')


def main():
    args = get_args()

    if not os.path.exists(os.path.join(project_root, args.model_path)):
        os.makedirs(os.path.join(project_root, args.model_path))

    with open(os.path.join(project_root, args.model_path, 'InputParm.json'), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    dtype = torch.float32
    dataset_file_info = get_all_foler_path(args.dataset_path)
    # trajecotry_file_info = get_all_trajectory_set_data(os.path.join(project_root, args.trajectory_path))
    osm_path = dataset_file_info[args.location_name]['map_file']
    envs = gym.make(
            args.env_name,
            render_mode="rgb_array",
            data_path=os.path.join(project_root, args.trajectory_path, args.location_name),
            osm_path=osm_path,
        )
    in_channels = 3
    action_dim = 5
    actor = CNNPolicy(in_channels, action_dim,)
    actor.to(dtype).to(device)

    critic = CNNValue(in_channels)
    critic.to(dtype).to(device)

    if args.algo == 'ppo':
        agent = algo.PPO(
            actor,
            critic,
            lr=args.lr,
            discount=args.gamma,
            actor_max_grad_norm=args.max_grad_norm,
            device=device,
        )
    else:
        log.error("Only PPO is supported")
        raise NotImplementedError

    if args.gail:
        discr_model = CNNDiscriminator(in_channels, action_dim)
        discr_model.to(dtype).to(device)
        discr = gail.Discriminator(discr_model, device)

        expert_dataset = gail.ExpertBEVDataset(os.path.join(project_root, args.gail_experts_dir))
        drop_last = len(expert_dataset) > args.gail_batch_size
        expert_data_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    for j in range(args.epochs):
        rollouts = Memory()
        num_steps = 0
        num_episodes = 0
        while num_steps < args.num_env_steps:
            num_episodes += 1
            state, _ = envs.reset()
            done = False
            while not done:
                # envs.render()
                with torch.no_grad():
                    state = expert_dataset.normalize_obs(state)
                    state = torch.from_numpy(state).to(dtype).to(device).unsqueeze(0)
                    actions = actor.select_action(state)
                    actions = to_device(torch.device('cpu'), *actions)
                    action = actions[0].cpu().numpy().squeeze()
                next_state, reward, terminated, truncated, info = envs.step(action)
                done = terminated or truncated
                mask = 0 if done else 1
                rollouts.push(state.cpu().numpy().squeeze(), action, mask, next_state, reward, actions)
                state = next_state
                num_steps += 1

        batch = rollouts.sample()
        # train
        next_states = torch.from_numpy(np.stack(batch.next_state)).to(dtype).to(device)
        states = np.stack(batch.state)
        actions = np.stack(batch.action)
        states = torch.from_numpy(states).to(dtype).to(device)
        actions = torch.from_numpy(actions).to(dtype).to(device)
        masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
        rewards = discr.predict_reward(states, actions)

        discr_log = discr.update(expert_data_loader, states, actions)
        agent_log = agent.update(states, actions, rewards, next_states, masks)

        log.info("Epoch: {}, "
                 "Discr fake loss: {},"
                 "Discr real loss: {},"
                 "Actor loss: {}, "
                 "Critic loss: {}".format(
            j,
            np.mean(discr_log['feak_discrim_loss']),
            np.mean(discr_log['real_discrim_loss']),
            np.mean(agent_log['actor_loss']),
            np.mean(agent_log['critic_loss'])
        ))

        if j % args.save_interval == 0:
            model_path = os.path.join(project_root, args.model_path, f"epoch_{j}.pth")
            torch.save(actor.state_dict(), model_path)

        save_exp_data(
            discr_log,
            agent_log,
            rewards.cpu().numpy(),
            num_episodes,
            num_steps,
            os.path.join(project_root, args.model_path),
            j,
            train=True,
        )


def save_exp_data(
    discr_log: dict,
    agent_log: dict,
    rewards: np.ndarray,
    num_episodes: int,
    num_steps: int,
    save_path: str,
    iter: int,
    train=True,
):
    csv_file_name = "train_loss.csv" if train else "eval_loss.csv"
    csv_file_path = os.path.join(save_path, csv_file_name)
    exp_data = [{
        'Discr fake loss': np.mean(discr_log['feak_discrim_loss']),
        'Discr real loss': np.mean(discr_log['real_discrim_loss']),
        'Actor loss': np.mean(agent_log['actor_loss']),
        'Critic loss': np.mean(agent_log['critic_loss']),
        'Rewards': np.mean(rewards),
        'Rewards std': np.std(rewards),
        'num_steps':num_steps,
        'num_episodes':num_episodes,
        'Epoch': iter,
    }]

    df = pd.DataFrame(exp_data)
    kwargs = {
        'index': False,
        'mode': 'w',
        'encoding': 'utf-8'
    }

    if iter > 0:
        kwargs['mode'] = 'a'
        kwargs['header'] = False

    df.to_csv(csv_file_path,**kwargs)


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gail',
        default=True,
        help='do imitation learning with gail')
    parser.add_argument('--dataset-path', type=str, default='', help="Interaction dataset path", required=True)
    parser.add_argument('--gail-experts-dir', type=str, default='data/dataset',
                        help='directory that contains expert demonstrations for gail')
    parser.add_argument('--location-name', type=str, default="DR_DEU_Merging_MT")
    parser.add_argument('--trajectory-path', type=str, default="data/trajectory_set")
    parser.add_argument('--model-path', type=str, default='data/model/gail', help="Experiment data path")
    parser.add_argument(
        '--epochs', type=int, default=2, help='gail epochs (default: 5)')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=64,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')

    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=40,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='save interval, one save per n updates (default: 1)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=1024,
        help='number of environment steps to train (default: 2048)')
    parser.add_argument(
        '--env-name',
        default='interaction-rl-v0',
        help='environment to train on (default: interaction-rl-v0)')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    assert args.algo in ['a2c', 'ppo', 'acktr']

    log.info("Running args:")
    for arg in vars(args):
        log.info("{}: {}".format(arg, getattr(args, arg)))
    log.info("---------------------------------------------------")

    return args


if __name__ == "__main__":
    main()