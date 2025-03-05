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
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 device=None):
        super(PPO, self).__init__()

        self.actor = actor
        self.critic = critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.v_optimizer = optim.Adam(critic.parameters(), lr=lr, eps=eps)
        self.p_optimizer = optim.Adam(actor.parameters(), lr=lr, eps=eps)
        self.device = device

    def update(self, states, actions, rewards, next_states,  masks):
        with torch.no_grad():
            # 估计状态价值
            values = self.critic(states)

        advantages, returns = estimate_advantages(rewards, masks, values, self.discount, self.tau, self.device)

        optim_iter_num = int(math.ceil(states.shape[0] / self.optim_batch_size))
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
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

                for _ in range(1):
                    values_pred = self.critic(states_b)  # 预测的值函数值
                    value_loss = (values_pred - returns_b).pow(2).mean()  # 计算值函数损失值
                    # 权重衰减
                    for param in self.critic.parameters():
                        value_loss += param.pow(2).sum() * self.l2_reg
                    self.v_optimizer.zero_grad()
                    value_loss.backward()
                    self.v_optimizer.step()

                log_probs = self.actor.get_log_prob(states_b, actions_b)
                ratio = torch.exp(log_probs - fixed_log_probs_b)
                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_b
                policy_surr = -torch.min(surr1, surr2).mean()
                self.p_optimizer.zero_grad()
                policy_surr.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.p_optimizer.step()

