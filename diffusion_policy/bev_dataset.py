import os
import sys
from typing import List

import torch
import numpy as np
import pickle
from simulation_environment.utils import lmap


class BEVDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_path: str,
            data_type: str = "pkl",
            train: bool = True,
    ):

        self.data_type = data_type
        self.dataset_path = dataset_path
        self.targets = np.load(os.path.join(dataset_path, 'targets.npy'))
        self.value_range = np.load(os.path.join(dataset_path, 'value_range.npy'))
        self.min = np.load(os.path.join(dataset_path, 'min.npy'))
        self.max = np.load(os.path.join(dataset_path, 'max.npy'))
        self.index = np.load(os.path.join(dataset_path, 'train_index.npy')) if train else np.load(os.path.join(dataset_path, 'eval_index.npy'))

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
        return len(self.index)

    def __getitem__(self, idx):
        # get the start/end indices

        target = self.targets[self.index[idx]]
        obs = pickle.load(open(os.path.join(self.dataset_path, f"{idx}.pkl"), 'rb'))
        normalize_obs = np.moveaxis(obs.astype(np.float32) / 255, -1, 0)
        nsample = dict()
        # discard unused observations
        nsample['image'] = normalize_obs
        nsample['action'] = target
        return nsample
