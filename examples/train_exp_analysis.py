import os
import sys
projector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(projector_dir)

from utils.font_path import chinese_font_path, english_font_path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from argparse import ArgumentParser


english_font = font_manager.FontProperties(fname=english_font_path, size=16)
chinese_font = font_manager.FontProperties(fname=chinese_font_path, size=16)


def plot_training_data(data_path, smooth=False, span=5, color=None):
    train_exp_data = pd.read_csv(os.path.join(projector_dir, data_path, 'train_loss.csv'))
    eval_exp_data = pd.read_csv(os.path.join(projector_dir, data_path, 'eval_loss.csv'))
    train_loss = train_exp_data['mean']
    eval_loss = eval_exp_data['mean']
    train_loss_std = train_exp_data['std'] * 0.5
    eval_loss_std = eval_exp_data['std'] * 0.5

    fig, ax = plt.subplots(1, 1)

    if smooth:
        train_loss = train_loss.ewma(span=span).mean()
        eval_loss = eval_loss.ewma(span=span).mean()
        train_loss_std = train_loss_std.ewma(span=span).mean()
        eval_loss_std = eval_loss_std.ewma(span=span).mean()
    # plot train loss curve
    x = train_exp_data['epoch'].to_numpy()
    ax.plot(x, train_loss, color=color[0][0], lw=2, label=f'Train', alpha=0.8, )
    ax.fill_between(x, train_loss - train_loss_std, train_loss + train_loss_std, color=color[0][1], alpha=0.8)
    # plot eval loss curve
    x = eval_exp_data['epoch'].to_numpy()
    ax.plot(x, eval_loss, color=color[1][0], lw=2, label=f'Eval', alpha=0.5)
    ax.fill_between(x, eval_loss - eval_loss_std, eval_loss + eval_loss_std, color=color[1][1], alpha=0.5)

    ax.set_ylabel('Loss', fontproperties=english_font)
    ax.set_xlabel('Epoch', fontproperties=english_font)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(prop=english_font)
    plt.tight_layout()
    plt.savefig(f'loss.svg', format='svg')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/model', help="Experiment data path")
    args = parser.parse_args()

    plot_training_data(args.data_path, color=[("#006400", "#8FBC8F"), ("#FF4500", "#FFA07A")])
