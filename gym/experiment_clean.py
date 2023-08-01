import json
from typing import NamedTuple

import os

import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import ray
from datetime import datetime
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer_clean import DecisionTransformer, DecisionTransformer_Product,\
    DecisionTransformer_Rotation
from decision_transformer.models.decision_transformer_clean_contrast import DecisionTransformerContrast, DecisionTransformerContrast_SIMCLR, \
    DecisionTransformerContrast_ProductSIMCLR

from decision_transformer.models.mlp_bc import MLPBCModel

from decision_transformer.training.seq_trainer_clean import SequenceTrainer, SequenceTrainerContrastive, SequenceTrainerContrastive_SIMCLR

from decision_transformer.utils.data import get_batch, get_digitized, get_bin_ranges, get_batch_contrastive, get_batch_contrastive_simclr
from decision_transformer.utils.data_class import DataClass, DataClassContrastive
from d4rl.hand_manipulation_suite.pen_v0 import PenEnvV0
from d4rl.hand_manipulation_suite.hammer_v0 import HammerEnvV0
from d4rl.hand_manipulation_suite.relocate_v0 import RelocateEnvV0


def discounted_cum_sum(rewards, discount = 0.99):
    returns = []
    accum = 0
    for r in rewards[::-1]:
        accum = r + discount*accum
        returns.append(accum)
    return np.array(returns)[::-1]

def get_returns(rewards, start_or_final, discount):
    end_indeces = np.argwhere(start_or_final == -1).flatten()
    start_indeces = np.argwhere(start_or_final == 1).flatten()
    g_t = np.zeros_like(rewards)
    for start, end in zip(start_indeces, end_indeces):
        if discount == 1:
            g_t[start:end+1] = np.cumsum(rewards[start:end+1][::-1])[::-1]
        else:
            g_t[start:end+1] = discounted_cum_sum(rewards[start:end+1], discount)
    return g_t


def prep_data(variant):
    if torch.cuda.is_available():
        device = variant.get('device', 'cuda')
    else:
        device = 'cpu'
    print(f'Using device: {device}')

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    mode = variant.get('mode', 'normal')

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600]
        scale = 1000.
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000]
        scale = 1000.
    elif env_name == 'pen':
        env = PenEnvV0()
        max_ep_len = 100
        env_targets = [10000]
        scale = 1000.
    elif env_name == 'hammer':
        env = HammerEnvV0()
        max_ep_len = 200
        env_targets = [17000]
        scale = 1000.
    elif env_name == 'relocate':
        env = RelocateEnvV0()
        max_ep_len = 200
        env_targets = [6000]
        scale = 1000.
    else:
        raise NotImplementedError

    assert len(env_targets) == 1, 'Error for parallel eval'

    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    embed_dim = variant['embed_dim']

    # load dataset
    if env_name in {'hopper', 'halfcheetah', 'walker2d'}:
        dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    else:
        dataset_path = f'data/{env_name}-{dataset}-v0.pkl'
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)

    # save all path information into separate lists
    states, traj_lens, path_returns, actions, rewards = [], [], [], [], []
    start_or_final = []
    state_to_path = []

    if model_type == 'dt_contrast_simclr' or 'dt_contrast_simclr_product':
        trajectories = list(filter(lambda x: len(x['observations']) > variant['num_samples_simclr'], trajectories))
    else:
        trajectories = list(filter(lambda x: len(x['observations']) > 1, trajectories))

    for path_ix, path in enumerate(trajectories):
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.

        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        path_returns.append(path['rewards'].sum())
        actions.append(path['actions'])
        rewards.append(path['rewards'])

        blank_arr = np.zeros((path['observations'].shape[0]), dtype=int)
        blank_arr[0] = 1
        blank_arr[-1] = -1
        start_or_final.append(blank_arr)
        state_to_path.append(np.ones(traj_lens[-1])*path_ix)

    traj_lens, path_returns = np.array(traj_lens), np.array(path_returns)
    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    normalized_states = (states - state_mean)/(state_std + 1e-5)
    start_or_final = np.concatenate(start_or_final, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    state_to_path = np.concatenate(state_to_path)
    num_timesteps = sum(traj_lens)
    g_t = get_returns(rewards, start_or_final, discount=1)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(path_returns):.2f}, std: {np.std(path_returns):.2f}')
    print(f'Max return: {np.max(path_returns):.2f}, min: {np.min(path_returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(path_returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2

    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    traj_inds = np.concatenate([np.argwhere(start_or_final == 1), np.argwhere(start_or_final == -1)], axis =-1)

    if model_type != 'dt_rot' and 'dt_contrast' not in model_type :
        return DataClass(states, normalized_states, state_mean, state_std, actions, rewards, g_t, start_or_final,
                         state_to_path, traj_inds,path_returns, traj_lens,p_sample , sorted_inds, state_dim, action_dim,
                         embed_dim, max_ep_len, scale, device, K, batch_size, env_targets, num_eval_episodes, mode), env
    else:
        num_bins = variant['num_bins']
        compress_dim = variant['compress_dim']
        bin_ranges = get_bin_ranges(g_t, num_bins, None if model_type != 'dt_rot' else max(env_targets))
        digitized = get_digitized(g_t, bin_ranges)
        unique_bins = np.sort(np.unique(digitized))
        return DataClassContrastive(
            states, normalized_states, state_mean, state_std, actions, rewards, g_t, start_or_final, state_to_path,
            traj_inds,path_returns, traj_lens,p_sample , sorted_inds, state_dim, action_dim, embed_dim, compress_dim,
            max_ep_len, scale, device, K, batch_size, env_targets, num_eval_episodes, mode, num_bins, digitized, bin_ranges,
            unique_bins, variant['num_samples_simclr']
        ), env

@ray.remote(num_cpus=0.1)
def eval_targets(model, data_class: DataClass, env, seed):
    if torch.cuda.is_available():
        model = model.to(data_class.device)
    env.seed(seed)
    target_rew = data_class.env_targets[0]
    returns, lengths = [], []
    for _ in range(data_class.num_eval_episodes):
        with torch.no_grad():
            if isinstance(model, DecisionTransformer) or isinstance(model, DecisionTransformerContrast):
                ret, length = evaluate_episode_rtg(
                    env,
                    data_class.state_dim,
                    data_class.action_dim,
                    model,
                    max_ep_len=data_class.max_ep_len,
                    scale=data_class.scale,
                    target_return=target_rew/data_class.scale,
                    mode=data_class.mode,
                    state_mean=np.array(data_class.state_mean),
                    state_std=np.array(data_class.state_std),
                    device=data_class.device,
                    data_class=data_class
                )
            else:
                ret, length = evaluate_episode(
                    env,
                    data_class.state_dim,
                    data_class.action_dim,
                    model,
                    max_ep_len=data_class.max_ep_len,
                    target_return=target_rew/data_class.scale,
                    mode=data_class.mode,
                    state_mean=data_class.state_mean,
                    state_std=data_class.state_std,
                    device=data_class.device,
                )
        returns.append(ret)
        lengths.append(length)

    return {
        'returns': returns, 'lengths': lengths
    }

def parallel_eval(model, data_class: DataClass, env):
    seeds = [1, 5, 10]
    model = model.to('cpu')
    model_ref = ray.put(model)
    data_cls_ref = ray.put(data_class)
    env_ref = ray.put(env)

    seed_data = ray.get([eval_targets.remote(model_ref, data_cls_ref, env_ref, seed) for seed in seeds])
    model.to(data_class.device)
    return {seeds[i]: seed_data[i] for i in range(len(seeds))}

def experiment(
        exp_prefix,
        variant,
):
    data_class, env = prep_data(variant)
    model_type = variant['model_type']

    ray.init(num_cpus = 3, _node_ip_address="0.0.0.0")

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=data_class.state_dim,
            act_dim=data_class.action_dim,
            max_length=data_class.K,
            max_ep_len=data_class.max_ep_len,
            hidden_size=data_class.embed_dim,
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*data_class.embed_dim,
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'dt_product':
        model = DecisionTransformer_Product(
            state_dim=data_class.state_dim,
            act_dim=data_class.action_dim,
            max_length=data_class.K,
            max_ep_len=data_class.max_ep_len,
            hidden_size=data_class.embed_dim,
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*data_class.embed_dim,
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'dt_rot':
        bins = (data_class.bin_ranges[1:] + np.roll(data_class.bin_ranges, 1)[1:])/(2*data_class.scale)
        std = np.ones_like(bins)
        model = DecisionTransformer_Rotation(
            state_dim=data_class.state_dim,
            act_dim=data_class.action_dim,
            max_length=data_class.K,
            max_ep_len=data_class.max_ep_len,
            hidden_size=data_class.embed_dim,
            bins= np.concatenate((bins.reshape(-1, 1), std.reshape(-1, 1)), axis=-1),
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*data_class.embed_dim,
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'dt_contrast_simclr':
        model = DecisionTransformerContrast_SIMCLR(
            state_dim=data_class.state_dim,
            act_dim=data_class.action_dim,
            max_length=data_class.K,
            max_ep_len=data_class.max_ep_len,
            hidden_size=data_class.embed_dim,
            compress_dim= 128, #128,
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*data_class.embed_dim,
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'dt_contrast_simclr_product':
        model = DecisionTransformerContrast_ProductSIMCLR(
            state_dim=data_class.state_dim,
            act_dim=data_class.action_dim,
            max_length=data_class.K,
            max_ep_len=data_class.max_ep_len,
            hidden_size=data_class.embed_dim,
            compress_dim=128,
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*data_class.embed_dim,
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=data_class.state_dim,
            act_dim=data_class.action_dim,
            max_length=data_class.K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=data_class.device)
    if isinstance(model, DecisionTransformer_Rotation):
        model.rotation = model.rotation.to(device=data_class.device)

    warmup_steps = variant['warmup_steps']
    if variant['pretrain']:
        optimizer = torch.optim.AdamW(
            [
                {'params': model.get_input_params(), 'lr': 0.00603217},
                {'params': model.get_non_input_params(), 'lr': variant['learning_rate']}
            ],
            weight_decay=variant['weight_decay'],
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=variant['learning_rate'],
            weight_decay=variant['weight_decay'],
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    log_to_wandb = variant.get('log_to_wandb', False)
    env_name, dataset = variant['env'], variant['dataset']
    name_experiment = variant['name']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{name_experiment}'

    if log_to_wandb:
        save_path = os.path.join('runs', env_name, dataset, name_experiment, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant.update({'save_path': save_path})
        )

    if model_type == 'dt' or model_type == 'dt_product' or model_type == 'dt_rot':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=data_class.batch_size,
            get_batch=get_batch,
            data_class = data_class,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            env = env,
            eval_fn=parallel_eval,
            log_to_wandb=log_to_wandb,
        )
    elif model_type == 'dt_contrast_simclr' or model_type == 'dt_contrast_simclr_product':
        trainer = SequenceTrainerContrastive_SIMCLR(
            model=model,
            optimizer=optimizer,
            batch_size=data_class.batch_size,
            get_batch=get_batch,
            get_batch_contrastive = get_batch_contrastive_simclr,
            data_class = data_class,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            env = env,
            eval_fn=parallel_eval,
            log_to_wandb=log_to_wandb,
            contrastive_frequency = variant['contrastive_frequency'],
            beta=variant['beta']
        )
    else:
        raise Exception(f'{model_type} not supported yet!')


    print(f'Starting iter')
    if variant['pretrain']:
        seeded_outputs = trainer.train_iteration(num_steps=100000, iter_num = 0, print_logs=True)
        optimizer.param_groups[0]['lr'] = variant['learning_rate']
    else:
        seeded_outputs = trainer.eval_method()

    if log_to_wandb:
        save_outputs(0, save_path, seeded_outputs)

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1,
                                          print_logs=True)
        if log_to_wandb:
            torch.save(model.state_dict(), os.path.join(save_path, f'dt_{iter}_model.pth'))
            save_outputs(iter, save_path, seeded_outputs)

def save_outputs(iter, save_path, outputs):
    run_save_path = os.path.join(save_path, 'run_data')
    if not os.path.exists(run_save_path):
        os.makedirs(run_save_path, exist_ok=True)


    with open(os.path.join(run_save_path, f'seeded_runs_{iter}.json'), "w") as outfile:
        json.dump(outputs, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--compress_dim', type=int, default=50)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', action='store_true', default=False,)
    parser.add_argument('--same_inds', action='store_true', default=False,)
    parser.add_argument('--name', type=str, default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    parser.add_argument('--num_bins', type=int, default=128)
    parser.add_argument('--contrastive_frequency', type=int, default=1)
    parser.add_argument('--num_samples_simclr', type=int, default=3)
    parser.add_argument('--pretrain', action='store_true', default=False,)
    parser.add_argument('--beta', type=float, default=0.1)

    args = parser.parse_args()
    experiment('gym-contrast_exp_parallel', variant=vars(args))
