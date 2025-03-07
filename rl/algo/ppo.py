import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def to_device(device, *args):
    return [x.to(device) for x in args]


def estimate_advantages(rewards, masks, values, gamma, tau, device):
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


class PPO():
    def __init__(self,
                 actor,
                 critic,
                 lr: float = 1e-3,
                 optim_epochs=10,
                 optim_batch_size=64,
                 discount: float = 0.99,
                 tau: float = 0.95,
                 l2_reg: float = 1e-3,
                 clip_epsilon: float = 0.2,
                 optima_value_iternum = 1,
                 actor_max_grad_norm: float = 40.,
                 device=None):
        super(PPO, self).__init__()

        self.actor = actor
        self.critic = critic
        self.optim_epochs = optim_epochs
        self.optim_batch_size = optim_batch_size
        self.discount = discount
        self.tau = tau
        self.l2_reg = l2_reg
        self.clip_epsilon = clip_epsilon
        self.optima_value_iternum = optima_value_iternum
        self.actor_max_grad_norm = actor_max_grad_norm
        self.v_optimizer = optim.Adam(critic.parameters(), lr=lr)
        self.p_optimizer = optim.Adam(actor.parameters(), lr=lr)
        self.device = device

    def update(self, states, actions, rewards, next_states,  masks):
        with torch.no_grad():
            values = self.critic(states)
            fixed_log_probs = self.actor.get_log_prob(states, actions)

        advantages, returns = estimate_advantages(rewards, masks, values, self.discount, self.tau, self.device)

        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
        log = {
            'actor_loss': [],
            'critic_loss': [],
        }

        for _ in range(self.optim_epochs):
            perm = np.arange(states.shape[0])
            # 对索引进行打乱
            np.random.shuffle(perm)
            # 将perm转换为long类型张量，并移动到指定设备
            perm = torch.LongTensor(perm).to(self.device)
            # 将数据通过索引的方式进行打乱
            states, actions, returns, advantages, fixed_log_probs = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone()

            for i in range(optim_iter_num):
                # 从打乱的数据采样小批量数据
                ind = slice(i * self.optim_batch_size, min((i + 1) * self.optim_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                    states[ind].to(self.device), actions[ind].to(self.device), advantages[ind].to(self.device), returns[ind].to(self.device), fixed_log_probs[ind].to(self.device)

                for _ in range(self.optima_value_iternum):
                    values_pred = self.critic(states_b)
                    value_loss = (values_pred - returns_b).pow(2).mean()
                    for param in self.critic.parameters():
                        value_loss += param.pow(2).sum() * self.l2_reg
                    self.v_optimizer.zero_grad()
                    value_loss.backward()
                    self.v_optimizer.step()
                    log['critic_loss'].append(value_loss.item())

                log_probs = self.actor.get_log_prob(states_b, actions_b)
                ratio = torch.exp(log_probs - fixed_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_b
                policy_surr = -torch.min(surr1, surr2).mean()
                self.p_optimizer.zero_grad()
                policy_surr.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_max_grad_norm)
                self.p_optimizer.step()
                log['actor_loss'].append(policy_surr.item())

        return log

