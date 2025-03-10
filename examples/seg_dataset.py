import os
import sys
projector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(projector_dir)

import subprocess
import pickle
import numpy as np
from simulation_environment.utils import lmap

if __name__ == '__main__':
    location_name = 'DR_DEU_Merging_MT'
    dataset_path = f"{projector_dir}/data/observation/{location_name}"
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
            targets.append([action['rel_position'][0], action['rel_position'][1], action['rel_velocity'][0],
                                 action['rel_velocity'][1], action['rel_yaw']])
            pickle.dump(obs, open(os.path.join(save_path, f"{count}.pkl"), 'wb'))
            count += 1

    targets = np.array(targets)
    orin_targets = targets.copy()
    min_value = np.min(targets, axis=0)
    max_value = np.max(targets, axis=0)
    value_range = np.concatenate((min_value, max_value), axis=0).reshape(2, -1)
    value_range = np.max(np.abs(value_range), axis=0)
    y = [-1, 1]
    targets = y[0] + (targets + value_range) * (y[1] - y[0]) / (2 * value_range)
    cal_orin_targets = -value_range + (targets - y[0]) * (2 * value_range) / (y[1] - y[0])
    targets = np.array(targets, dtype=np.float32)

    np.save(f"{save_path}/min.npy", min_value)
    np.save(f"{save_path}/max.npy", max_value)
    np.save(f"{save_path}/value_range.npy", value_range)
    np.save(f"{save_path}/targets.npy", targets)

    total_index = np.arange(len(targets))
    np.random.shuffle(total_index)
    train_size = int(len(targets)*0.8)
    eval_size = len(targets) - train_size

    train_index = total_index[:train_size]
    eval_index = total_index[train_size:]

    np.save(f"{save_path}/train_index.npy", train_index)
    np.save(f"{save_path}/eval_index.npy", eval_index)

    print(f"loss: {np.linalg.norm(cal_orin_targets - orin_targets)}")
    print(f"min: {min_value}")
    print(f"max: {max_value}")
    print(f"value_range: {value_range}")
