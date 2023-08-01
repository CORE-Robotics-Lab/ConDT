import csv
import logging
# make deterministic
import sys
from datetime import datetime

import h5py

from mingpt.utils import set_seed, get_bin_ranges, get_digitized
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig, GPT_Product, GPT_Product_SIMCLR, GPT_SIMCLR, GPT_Concat, \
    GPT_Concat_SIMCLR, GPT_Rotation
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import argparse
from create_dataset import create_dataset
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
parser.add_argument('--model', type=str, default='dt')
parser.add_argument('--log_to_wandb', '-w', action='store_true', default=False,)
parser.add_argument('--scaled', action='store_true', default=False,)
parser.add_argument('--pretrain', action='store_true', default=False,)
parser.add_argument('--num_bins', type=int, default=128,)

args = parser.parse_args()

set_seed(args.seed)

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

    def get_contrastive(self, idx, block_size):
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)

        return states, actions, rtgs

if os.path.exists(os.path.join(args.data_dir_prefix, args.game, 'recorded_data.h5')) and args.num_steps > 1000:
    print('Reading from file!')
    with h5py.File(os.path.join(args.data_dir_prefix, args.game, 'recorded_data.h5'), 'r') as f:
        obss = f['obs'][:]
        actions = f['actions'][:]
        returns = f['returns'][:]
        done_idxs = f['dones'][:]
        rtgs = f['rtgs'][:]
        timesteps = f['timesteps'][:]
else:
    print('Creating Dataset!')
    obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game,
                                                                        args.data_dir_prefix, args.trajectories_per_buffer)
    np_obss = np.concatenate([obs[np.newaxis, ...] for obs in obss], axis=0)

    with h5py.File(os.path.join(args.data_dir_prefix, args.game, 'recorded_data.h5'), 'w') as hf:
        hf.create_dataset('obs', data=np_obss, compression='gzip', compression_opts=9,)
        hf.create_dataset('actions', data=actions)
        hf.create_dataset('returns', data=returns)
        hf.create_dataset('dones', data=done_idxs)
        hf.create_dataset('rtgs', data=rtgs)
        hf.create_dataset('timesteps', data=timesteps)


# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)

if args.game == 'Breakout':
    target = 90
elif args.game == 'Seaquest':
    target = 1150
elif args.game == 'Qbert':
    target = 14000
elif args.game == 'Pong':
    target = 20
else:
    raise NotImplementedError()

bin_ranges = get_bin_ranges(train_dataset.rtgs, args.num_bins, target)

bins = (bin_ranges[1:] + np.roll(bin_ranges, 1)[1:])/2
std = np.ones_like(bins)*10

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type,
                  max_timestep=max(timesteps), compress_dim=64 if args.game != 'Pong' else 256,
                  num_bins=args.num_bins, bins=np.concatenate((bins.reshape(-1, 1), std.reshape(-1, 1)), axis=-1))

if args.model == 'dt':
    print('Using DT')
    model = GPT(mconf)
elif args.model == 'dt_product':
    print('Using DTProduct')
    model = GPT_Product(mconf)
elif args.model == 'dt_contrast':
    print('Using DT with SIMCLR')
    model = GPT_SIMCLR(mconf)
elif args.model == 'dt_contrast_product':
    print('Using DTProduct with SIMCLR')
    model = GPT_Product_SIMCLR(mconf)
elif args.model == 'dt_rot':
    print('using DT Rotation')
    model = GPT_Rotation(mconf)
else:
    raise Exception(f'Havent implemented model {args.model}')

# initialize a trainer instance and kick off training

rtg_scale = 1
if isinstance(model, GPT_Product) and args.scaled:
    if args.game == 'Breakout':
        rtg_scale = 100
    elif args.game == 'Seaquest':
        rtg_scale = 1500
    elif args.game == 'Qbert':
        rtg_scale = 15000
    elif args.game == 'Pong':
        rtg_scale = 50
    else:
        raise NotImplementedError()

save_path = os.path.join('runs', args.game, args.model, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

epochs = args.epochs
tconf = TrainerConfig(
    max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
    lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
    num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps),
    rtg_scale = rtg_scale, log_to_wandb=args.log_to_wandb, name=args.model,
    save_path = save_path, scaled = args.scaled, pretrain = args.pretrain,
    target = target, bin_ranges=bin_ranges
)
trainer = Trainer(model, train_dataset, None, tconf)
trainer.train(contrastive=isinstance(model, GPT_SIMCLR), pretrain=args.pretrain)
