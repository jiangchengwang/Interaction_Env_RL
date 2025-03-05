import os
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from rl.networks import ResNet10
import pickle


class Discriminator(object):
    def __init__(self, model, device=None):
        super(Discriminator, self).__init__()
        self.device = device
        self.discri = model.to(device)
        self.discri.train()
        self.discrim_criterion = torch.nn.BCELoss()
        self.discrim_criterion.to(device)
        self.optimizer = torch.optim.Adam(self.discri.parameters())
        self.returns = None

    def update(self,
               expert_data_loader,
               policy_states,
               policy_actions):
        for nbatch in expert_data_loader:
            expert_states = nbatch['image'].to(policy_states.dtype).to(self.device)
            expert_actions = nbatch['action'].to(policy_states.dtype).to(self.device)
            g_o = self.discri(policy_states, policy_actions)
            e_o = self.discri(expert_states, expert_actions)
            self.optimizer.zero_grad()
            feak_discrim_loss = self.discrim_criterion(g_o, torch.ones((policy_states.shape[0], 1), device=self.device))
            real_discrim_loss = self.discrim_criterion(e_o, torch.zeros((expert_states.shape[0], 1), device=self.device))
            discrim_loss = feak_discrim_loss + real_discrim_loss
            discrim_loss.backward()
            self.optimizer.step()

    def predict_reward(self, states, actions):
        with torch.no_grad():
            d = self.discri(states, actions)
            reward = -torch.log(d).squeeze()
            return reward


class ExpertBEVDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_path: str,
            data_type: str = "pkl",
    ):

        self.data_type = data_type
        self.dataset_path = dataset_path
        self.targets = np.load(os.path.join(dataset_path, 'targets.npy'))
        self.value_range = np.load(os.path.join(dataset_path, 'value_range.npy'))
        self.min = np.load(os.path.join(dataset_path, 'min.npy'))
        self.max = np.load(os.path.join(dataset_path, 'max.npy'))

    def normalize_data(self, ndata, y: List=[-1, 1]):
        # nomalize to [0,1]
        ndata = y[0] + (ndata + self.value_range) * (y[1] - y[0]) / (2 * self.value_range)
        return ndata

    def unnormalize_data(self, ndata, y: List=[-1, 1]):
        ndata = -self.value_range + (ndata - y[0]) * (2 * self.value_range) / (y[1] - y[0])
        return ndata

    def normalize_obs(self, obs):
        return np.moveaxis(obs.astype(np.float32) / 255, -1, 0)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        target = self.targets[idx]
        obs = pickle.load(open(os.path.join(self.dataset_path, f"{idx}.pkl"), 'rb'))
        normalize_obs = np.moveaxis(obs.astype(np.float32) / 255, -1, 0)
        nsample = dict()
        nsample['image'] = normalize_obs
        nsample['action'] = target
        return nsample
