import csv
import logging
# make deterministic
import numpy as np
import os
import h5py

from mingpt.utils import set_seed
import argparse
from create_dataset import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
args = parser.parse_args()

set_seed(args.seed)
obss_list, np_actions, np_returns, np_done_idxs, np_rtgs, np_timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)
np_obss = np.concatenate([obs[np.newaxis, ...] for obs in obss_list], axis=0)

hf = h5py.File(os.path.join(args.data_dir_prefix, args.game, 'recorded_data.h5'), 'w')
hf.create_dataset('obs', data=np_obss, compression='gzip', compression_opts=9,)
hf.create_dataset('actions', data=np_actions)
hf.create_dataset('returns', data=np_returns)
hf.create_dataset('dones', data=np_done_idxs)
hf.create_dataset('rtgs', data=np_rtgs)
hf.create_dataset('timesteps', data=np_timesteps)
hf.close()