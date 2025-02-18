import os
import sys
projector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(projector_dir)

import subprocess
import pickle
import numpy as np


if __name__ == '__main__':
    dataset_path = f"{projector_dir}/data/observation/DR_CHN_Roundabout_LN"
    save_path = f"{projector_dir}/data/dataset"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # read dataset
    res = subprocess.getoutput(f"find {dataset_path} -name *.pkl")
    dataset_list = res.split("\n")

    count = 0
    targets = []
    for dp in dataset_list:
        scenario_data = pickle.load(open(dp, 'rb'))
        for i in range(len(scenario_data['obs'])):
            obs = scenario_data['obs'][i]
            action = scenario_data['action'][i]
            targets.append([action['rel_position'][0], action['rel_position'][1], action['rel_velocity'][0][0],
                                 action['rel_velocity'][1][0], action['rel_yaw']])
            pickle.dump(obs, open(os.path.join(save_path, f"{count}.pkl"), 'wb'))
            count += 1

    targets = np.array(targets)
    min_value = np.min(targets, axis=0)
    max_value = np.max(targets, axis=0)
    targets = (targets - min_value) / (max_value - min_value) * 2 - 1
    targets = targets.astype(np.float32)

    np.save(f"{save_path}/min.npy", min_value)
    np.save(f"{save_path}/max.npy", max_value)
    np.save(f"{save_path}/targets.npy", targets)

    total_index = np.arange(len(targets))
    np.random.shuffle(total_index)
    train_size = int(len(targets)*0.8)
    eval_size = len(targets) - train_size

    train_index = total_index[:train_size]
    eval_index = total_index[train_size:]

    np.save(f"{save_path}/train_index.npy", train_index)
    np.save(f"{save_path}/eval_index.npy", eval_index)
