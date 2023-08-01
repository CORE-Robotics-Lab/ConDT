import os

import random

import h5py
import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.optim as optim
from pytorch_metric_learning.losses import NTXentLoss
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import os
import platform

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

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


class ContrastiveModuleRCRCL_simclr(nn.Module):
    def __init__(self, n_embd, vocab_size, compress_dim):
        super().__init__()
        self.n_embd = n_embd
        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, n_embd), nn.Tanh()
        )

        self.ret_emb = nn.Sequential(nn.Linear(1, n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(vocab_size, n_embd), nn.Tanh())
        self.compress = nn.Linear(2*n_embd, compress_dim)

    def get_latent(self, s, a):
        s_encoded = self.state_encoder(s.reshape(-1, 4, 84, 84).type(torch.float32).contiguous())
        s_encoded = s_encoded.reshape(s.shape[0], s.shape[1], self.n_embd)
        a_encoded = self.action_embeddings(a.type(torch.long).squeeze(-1))

        return self.compress(torch.cat([s_encoded, a_encoded], axis=-1))

def get_loss(
        model,
        states: torch.Tensor,
        actions: torch.Tensor,
        rtg: torch.Tensor,
        device: torch.device,
        temperature: float = 0.5,
        num_positives: int = 2,
        use_simclr=True
):
    B, N, state_dim = states.shape
    _, _, action_dim = actions.shape

    ixs = np.concatenate([np.random.choice(np.arange(N), size=num_positives).reshape(1, -1) for _ in range(B)], axis=0)
    torch_ixs = torch.from_numpy(ixs).to(device=device)
    if use_simclr:
        pos_states = torch.cat([states[b_ix, torch_ixs[b_ix]].unsqueeze(0) for b_ix in range(B)])
        pos_actions = torch.cat([actions[b_ix, torch_ixs[b_ix]].unsqueeze(0) for b_ix in range(B)])
        pos_rtg = torch.cat([rtg[b_ix, torch_ixs[b_ix]].unsqueeze(0) for b_ix in range(B)])

        pos_latents = model.get_latent(pos_states, pos_actions)
        proj_out_dim = pos_latents.shape[-1]

        labels = torch.arange(0, B).reshape(-1, 1).repeat(1, num_positives).flatten().to(device)
        simclr_loss = NTXentLoss(temperature=temperature)
        loss = simclr_loss(embeddings = pos_latents.reshape(-1, proj_out_dim), labels=labels)
        return loss
    else:
        raise Exception('hasnt been handled yet')

def get_all_data(context_length):
    env_name = 'Pong'

    if 'mac' in platform.platform():
        path = '/Users/sachinkonan/Documents/Research/traj_opt/DTTrajectoryOpt/atari'
    else:
        path = '/nethome/skonan3/DTTrajectoryOpt/atari'

    with h5py.File(os.path.join(path, 'dqn_replay', env_name, 'recorded_data.h5'), 'r') as f:
        obss = f['obs'][:]
        actions = f['actions'][:]
        returns = f['returns'][:]
        done_idxs = f['dones'][:]
        rtgs = f['rtgs'][:]
        timesteps = f['timesteps'][:]

    return StateActionReturnDataset(obss, context_length*3, actions, done_idxs, rtgs, timesteps)

def train_rcrl(config):
    train_data = get_all_data(config['context_length'])
    net = ContrastiveModuleRCRCL_simclr(128, train_data.vocab_size, compress_dim=config['compress_dim'])
    optimizer = optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=0.1)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    net = net.to(device)

    max_iters = 5000
    val_iters = 100
    max_epochs = 10

    batch_size = config['batch_size']
    temperature = config['temperature']
    num_positives = config['num_positives']

    loader = DataLoader(
        train_data, shuffle=True, pin_memory=True,
        batch_size=batch_size, num_workers=0
    )

    net.eval()
    start_val_loss = 0
    for ix, (s,a,rtg,t) in enumerate(loader):
        if ix >= val_iters:
            break

        s = s.to(device)
        a = a.to(device)
        rtg = rtg.to(device)
        t = t.to(device)

        loss = get_loss(net, s, a, rtg, device, num_positives=num_positives, temperature=temperature)
        start_val_loss += loss

    start_val_loss = (start_val_loss/val_iters).clone().detach().cpu().numpy()

    for epoch in range(max_epochs):
        net.train()
        for ix, (s,a,rtg,t) in enumerate(loader):
            if ix >= max_iters:
                break

            s = s.to(device)
            a = a.to(device)
            rtg = rtg.to(device)
            t = t.to(device)

            loss = get_loss(net, s, a, rtg, device, num_positives=num_positives, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), .25)
            optimizer.step()

        net.eval()
        val_loss = 0
        for ix, (s,a,rtg,t) in enumerate(loader):
            if ix >= val_iters:
                break

            s = s.to(device)
            a = a.to(device)
            rtg = rtg.to(device)
            t = t.to(device)

            loss = get_loss(net, s, a, rtg, device, num_positives=num_positives, temperature=temperature)
            val_loss += loss
        val_loss = val_loss.clone().detach().cpu().numpy()/val_iters

        #print(dict(percent_change=(start_val_loss - val_loss)/start_val_loss, loss=val_loss, original_loss=start_val_loss))
        tune.report(percent_change=(start_val_loss - val_loss)/start_val_loss, loss=val_loss, original_loss=start_val_loss)
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path)

def hyperopt_search(num_samples = 10, gpus_per_trial: float = 0):
    config = {
        "compress_dim": tune.sample_from(lambda _: 2**np.random.randint(5, 9)),
        "batch_size": tune.choice([64]),
        "lr": tune.loguniform(1e-5, 1e-2),
        "temperature": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        "num_positives": tune.choice([2, 3]),
        "context_length": tune.choice([15, 30, 50])
    }

    max_num_epochs = 10

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )

    result = tune.run(
        tune.with_parameters(train_rcrl),
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        config=config,
        metric="percent_change",
        mode="max",
        num_samples=num_samples,
        scheduler=scheduler
    )

    df_results = result.results_df
    df_results.to_csv('results_3.csv')

    best_trial = result.get_best_trial("percent_change", "max", "last")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final % change: {}".format(
        best_trial.last_result["percent_change"]))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

hyperopt_search(10, 0.5 if torch.cuda.is_available() else 0)
#train_rcrl({'batch_size': 64, 'lr': 00737755, 'temperature': 0.1, 'num_positives': 3, 'compress_dim': 128})
#train_rcrl({'batch_size': 64, 'lr': 5e-4, 'temperature': 0.1, 'num_positives': 3, 'compress_dim': 128})
