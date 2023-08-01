import numpy as np
import torch
from typing import NamedTuple


class DataClass(NamedTuple):
    """Represents Data"""
    states: np.ndarray # samples x state_dim
    normalized_states: np.ndarray # samples x state_dim
    state_mean: np.ndarray
    state_std: np.ndarray
    actions: np.ndarray # samples x action_dim
    rewards: np.ndarray # samples x 1
    rtg: np.ndarray # samples x 1

    start_or_final: np.ndarray # samples x 1
    state_to_path: np.ndarray # samples x 1

    glob_start_end_indeces: np.ndarray # num_trajectoresx2
    path_returns: np.ndarray # num_trajectories x 1
    traj_lens: np.ndarray # num_trajectories x 1
    p_sample: np.ndarray # num_trajectories x 1
    sorted_inds: np.ndarray # num_trajectories x 1

    state_dim: int
    action_dim: int
    embed_dim: int

    max_ep_len: int
    scale: float
    device: torch.device

    K: float
    batch_size: int

    env_targets: list
    num_eval_episodes: int
    mode: str

class DataClassContrastive(NamedTuple):
    """Represents Data"""
    states: np.ndarray # samples x state_dim
    normalized_states: np.ndarray # samples x state_dim
    state_mean: np.ndarray
    state_std: np.ndarray
    actions: np.ndarray # samples x action_dim
    rewards: np.ndarray # samples x 1
    rtg: np.ndarray # samples x 1

    start_or_final: np.ndarray # samples x 1
    state_to_path: np.ndarray # samples x 1

    glob_start_end_indeces: np.ndarray # num_trajectoresx2
    path_returns: np.ndarray # num_trajectories x 1
    traj_lens: np.ndarray # num_trajectories x 1
    p_sample: np.ndarray # num_trajectories x 1
    sorted_inds: np.ndarray # num_trajectories x 1

    state_dim: int
    action_dim: int
    embed_dim: int
    compress_dim:int

    max_ep_len: int
    scale: float
    device: torch.device

    K: float
    batch_size: int

    env_targets: list
    num_eval_episodes: int
    mode: str

    num_bins: int
    digitized: np.ndarray
    bin_ranges: np.ndarray
    unique_bins: np.ndarray

    num_samples_simclr: int