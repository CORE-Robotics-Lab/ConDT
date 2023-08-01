import os

import random
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

state_dim = 11
action_dim = 3
embed_dim = 128

def discounted_cum_sum(rewards, discount = 0.99):
    returns = []
    accum = 0
    for r in rewards[::-1]:
        accum = r + discount*accum
        returns.append(accum)
    return np.array(returns)[::-1]

def get_segments(trajectories, state_mean, state_std, device, batch_size=256, max_len=20, num_samples = 2):
    batch_inds = np.random.choice(
        np.arange(len(trajectories)),
        size=batch_size,
        replace=True,
    )

    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        traj = trajectories[batch_inds[i]]
        si = random.randint(0, traj['rewards'].shape[0] - 1 - num_samples)

        # get sequences from dataset
        s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
        a.append(traj['actions'][si:si + max_len].reshape(1, -1, action_dim))
        r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
        rtg.append(discounted_cum_sum(traj['rewards'][si:], discount=1.)[:s[-1].shape[1]].reshape(1, -1, 1))

        # padding and state + reward normalization
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate([np.ones((1, max_len - tlen, action_dim)) * -10., a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / 1000

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)

    return s, a, rtg

class ContrastiveModuleRCRCL_simclr(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim, compress_dim):
        super().__init__()
        self.embed_state = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, compress_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.embed_action = nn.Sequential(
            nn.Linear(act_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, compress_dim),
            nn.LeakyReLU(inplace=True)
        )

        self.compress = nn.Linear(2*compress_dim, compress_dim)

    def get_latent(self, states, action, rtg):
        s_a = torch.cat([self.embed_state(states), self.embed_action(action)], dim=-1)
        return self.compress(s_a)

class ContrastiveModuleRCRCL_simclr_v2(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim, compress_dim):
        super().__init__()
        self.embed_state = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.embed_action = nn.Sequential(
            nn.Linear(act_dim, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.embed_rtg = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(inplace=True),
        )

        self.compress = nn.Linear(2*hidden_dim, compress_dim)

    def get_latent(self, states, action, rtg):
        embedded_rtg = self.embed_rtg(rtg)
        s_a = torch.cat([self.embed_state(states)*embedded_rtg,
                         self.embed_action(action)*embedded_rtg], dim=-1)
        return self.compress(s_a)

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

        pos_latents = model.get_latent(pos_states, pos_actions, pos_rtg)
        proj_out_dim = pos_latents.shape[-1]

        labels = torch.arange(0, B).reshape(-1, 1).repeat(1, num_positives).flatten().to(device)
        simclr_loss = NTXentLoss(temperature=temperature)
        loss = simclr_loss(embeddings = pos_latents.reshape(-1, proj_out_dim), labels=labels)
        return loss
    else:
        raise Exception('hasnt been handled yet')

def get_all_data():
    env_name = 'hopper'
    dataset = 'medium'

    if 'mac' in platform.platform():
        path = '~/Documents/Research/traj_opt/DTTrajectoryOpt/gym'
    else:
        path = '/nethome/_/DTTrajectoryOpt/gym'

    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(os.path.join(path, dataset_path), 'rb') as f:
        trajectories = pickle.load(f)

    states = []
    for path_ix, path in enumerate(trajectories):
        states.append(path['observations'])
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    return trajectories, state_mean, state_std

def train_rcrl(config):
    #net = ContrastiveModuleRCRCL_simclr(state_dim, action_dim, hidden_dim=128, compress_dim=config['compress_dim'])
    net = ContrastiveModuleRCRCL_simclr_v2(state_dim, action_dim, hidden_dim=128, compress_dim=config['compress_dim'])
    optimizer = optim.AdamW(net.parameters(), lr=config['lr'], weight_decay=1e-4)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    net.to(device)

    max_iters = 1000
    val_iters = 100
    max_epochs = 10

    trajectories, s_mean, s_std = get_all_data()

    batch_size = config['batch_size']
    temperature = config['temperature']
    num_positives = config['num_positives']

    net.eval()
    start_val_loss = 0
    for ix in range(0, val_iters):
        s, a, r = get_segments(trajectories, s_mean, s_std, device, num_samples=num_positives, batch_size=batch_size)
        loss = get_loss(net, s, a, r, device, num_positives=num_positives, temperature=temperature)
        start_val_loss += loss
    start_val_loss = (start_val_loss/val_iters).clone().detach().cpu().numpy()

    for epoch in range(max_epochs):
        net.train()
        for ix in range(0, max_iters):
            s, a, r = get_segments(trajectories, s_mean, s_std, device, num_samples=num_positives, batch_size=batch_size)
            loss = get_loss(net, s, a, r, device, num_positives=num_positives, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), .25)
            optimizer.step()

        net.eval()
        val_loss = 0
        for ix in range(0, val_iters):
            s, a, r = get_segments(trajectories, s_mean, s_std, device, num_samples=num_positives, batch_size=batch_size)
            loss = get_loss(net, s, a, r, device, num_positives=num_positives, temperature=temperature)
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
        "temperature": tune.choice([0.1]),
        "num_positives": tune.choice([2, 3])
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
    df_results.to_csv('results_2.csv')

    best_trial = result.get_best_trial("percent_change", "max", "last")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final % change: {}".format(
        best_trial.last_result["percent_change"]))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))

hyperopt_search(10, 0.5 if torch.cuda.is_available() else 0)
#train_rcrl({'batch_size': 64, 'lr': 00737755, 'temperature': 0.1, 'num_positives': 3, 'compress_dim': 128})
#train_rcrl({'batch_size': 64, 'lr': 5e-4, 'temperature': 0.1, 'num_positives': 3, 'compress_dim': 128})
