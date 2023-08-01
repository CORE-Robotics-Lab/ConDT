import numpy as np
import random
import torch
from decision_transformer.utils.data_class import DataClass, DataClassContrastive

def anchor_inds(data_class: DataClass):
    batch_size = data_class.batch_size
    num_trajectories = len(data_class.traj_lens)
    p_sample = data_class.p_sample
    sorted_inds = data_class.sorted_inds

    batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
        p=p_sample,  # reweights so we sample according to timesteps
    )

    glob_start_end_indeces = data_class.glob_start_end_indeces
    inds = []
    for i in range(batch_size):
        traj_ix = int(sorted_inds[batch_inds[i]])
        glob_start_ix, glob_end_ix = glob_start_end_indeces[traj_ix]
        si = random.randint(glob_start_ix, glob_end_ix)
        inds.append([si, glob_start_ix, glob_end_ix])
    return np.array(inds)

def get_batch(data_class: DataClass, inds=None):
    batch_size = data_class.batch_size
    num_trajectories = len(data_class.traj_lens)
    p_sample = data_class.p_sample
    sorted_inds = data_class.sorted_inds
    if inds is not None:
        assert len(inds) == batch_size, 'Inds must be batch_size long'

    batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

    max_ep_len = data_class.max_ep_len
    max_len = data_class.K
    scale = data_class.scale
    device = data_class.device
    state_dim = data_class.state_dim
    action_dim = data_class.action_dim

    normalized_states = data_class.normalized_states
    actions = data_class.actions
    g_t = data_class.rtg
    glob_start_end_indeces = data_class.glob_start_end_indeces

    s, a, rtg, timesteps, mask, path_inds = [], [], [], [], [], []
    for i in range(batch_size):
        if inds is not None:
            si, glob_start_ix, glob_end_ix  = inds[i]
        else:
            traj_ix = int(sorted_inds[batch_inds[i]])
            glob_start_ix, glob_end_ix = glob_start_end_indeces[traj_ix]
            si = random.randint(glob_start_ix, glob_end_ix)
        end_ix = min(si+max_len, glob_end_ix + 1)
        tlen = end_ix - si

        states_per_traj = normalized_states[si:end_ix].reshape(1, -1, state_dim)
        actions_per_traj = actions[si:end_ix].reshape(1, -1, action_dim)
        timesteps_per_traj = np.arange(si - glob_start_ix, si - glob_start_ix + tlen).reshape(1, -1)
        timesteps_per_traj[timesteps_per_traj >= max_ep_len] = max_ep_len - 1  # padding cutoff
        rtg_per_traj = g_t[si:end_ix].reshape(1, -1, 1)

        s.append(np.concatenate([np.zeros((1, max_len - tlen, state_dim)), states_per_traj], axis=1))
        a.append(np.concatenate([np.ones((1, max_len - tlen, action_dim)) * -10., actions_per_traj], axis=1))
        rtg.append(np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg_per_traj], axis=1) / scale)
        timesteps.append(np.concatenate([np.zeros((1, max_len - tlen)), timesteps_per_traj], axis=1))
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
        path_inds.append(np.concatenate([np.ones((1, max_len - tlen)) * - 1, np.arange(si, end_ix).reshape(1, -1)], axis=1))

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
    path_inds = torch.from_numpy(np.concatenate(path_inds, axis=0)).to(device=device, dtype=torch.long)

    return s, a, rtg, timesteps, mask, path_inds

def get_bin_ranges(g_t, num_bins, max_val=None):
    max_g_t = max_val if max_val is not None else g_t.max()
    bins = np.linspace(0, max_g_t + 1, num_bins + 1, dtype=int)
    return bins

def get_digitized(g_t, bin_ranges):
    digitized = np.digitize(g_t, bin_ranges)
    return digitized

def get_batch_contrastive(data_class: DataClassContrastive, inds=None):
    digitized = data_class.digitized
    batch_size = data_class.batch_size
    normalized_states = data_class.normalized_states
    actions = data_class.actions
    rtg = data_class.rtg
    state_dim = data_class.state_dim
    action_dim = data_class.action_dim
    device = data_class.device
    all_bins = data_class.unique_bins
    scale = data_class.scale

    if inds is not None:
        assert len(inds) == batch_size, 'Error here'
        batch_bins = digitized[inds[:, 0]]
        neg_batch_bins = np.array([np.random.choice(all_bins[all_bins != b_t], 1)[0] for b_t in batch_bins])
        labels = np.concatenate([batch_bins.reshape(-1, 1), neg_batch_bins.reshape(-1, 1)], axis=-1)
        anchors = inds[:, [0]]
        all_inds = np.arange(0, len(digitized))
        pos = []
        for ix in range(0, len(batch_bins)):
            b_t = batch_bins[ix]
            anchor_ind = anchors[ix]
            mask = np.bitwise_and(digitized == b_t, all_inds != anchor_ind)
            pos.append(np.random.choice(np.argwhere(mask).flatten(), 1))
        anchor_pos = np.concatenate((anchors, pos), axis=-1)
        neg = np.array([np.random.choice(np.argwhere(digitized == b_t).flatten(), 1) for b_t in neg_batch_bins])
    else:
        batch_bins = np.random.choice(all_bins, batch_size)
        neg_batch_bins = np.array([np.random.choice(all_bins[all_bins != b_t], 1)[0] for b_t in batch_bins])
        labels = np.concatenate([batch_bins.reshape(-1, 1), neg_batch_bins.reshape(-1, 1)], axis=-1)

        anchor_pos = np.array([np.random.choice(np.argwhere(digitized == b_t).flatten(), 2) for b_t in batch_bins])
        neg = np.array([np.random.choice(np.argwhere(digitized == b_t).flatten(), 1) for b_t in neg_batch_bins])
    anchor_pos_neg = np.concatenate((anchor_pos, neg), axis=-1)

    sample_states = normalized_states[anchor_pos_neg.flatten()].reshape(anchor_pos_neg.shape + (state_dim, ))
    sample_actions = actions[anchor_pos_neg.flatten()].reshape(anchor_pos_neg.shape + (action_dim, ))
    sample_rtg = rtg[anchor_pos_neg.flatten()].reshape(anchor_pos_neg.shape)

    torch_states = torch.from_numpy(sample_states).to(dtype=torch.float32, device=device)
    torch_actions = torch.from_numpy(sample_actions).to(dtype=torch.float32, device=device)
    torch_rtg = torch.from_numpy(sample_rtg/scale).to(dtype=torch.float32, device=device)
    torch_labels = torch.from_numpy(labels).to(dtype=torch.int, device=device)

    return torch_states, torch_actions, torch_rtg, torch_labels

def get_pos(batch_bins, anchor, digitized, state_to_path):
    pos = []
    for b_t, anch in zip(batch_bins, anchor):
        digitized_mask = digitized == b_t
        same_seg = np.argwhere(np.bitwise_and(digitized_mask, state_to_path == state_to_path[anch[0]]))
        diff_seg = np.argwhere(np.bitwise_and(digitized_mask, state_to_path != state_to_path[anch[0]]))
        inds = np.argwhere(digitized_mask).flatten()
        p = np.zeros_like(inds, dtype=np.float64)
        p[np.isin(inds, same_seg)] = 1.0
        p[np.isin(inds, diff_seg)] = 2.0
        p[inds == anch[0]] = 0.0 # we don't want to pick the anchor as the pos sample
        p /= p.sum()
        pos.append(np.random.choice(inds, 1, p=p))
    return pos

def get_batch_contrastive_hard(data_class: DataClassContrastive):
    digitized = data_class.digitized
    min_bin, max_bin = digitized.min(), digitized.max()
    batch_size = data_class.batch_size
    normalized_states = data_class.normalized_states
    actions = data_class.actions
    state_dim = data_class.state_dim
    action_dim = data_class.action_dim
    device = data_class.device

    all_bins = np.arange(min_bin, max_bin + 1)
    batch_bins = np.random.choice(all_bins, batch_size)
    neg_batch_bins = np.array([np.random.choice(all_bins[all_bins != b_t], 1)[0] for b_t in batch_bins])
    labels = np.concatenate([batch_bins.reshape(-1, 1), neg_batch_bins.reshape(-1, 1)], axis=-1)

    anchor = np.array([np.random.choice(np.argwhere(digitized == b_t).flatten(), 1) for b_t in batch_bins])
    pos = get_pos(batch_bins, anchor, data_class.digitized, data_class.state_to_path)
    neg = np.array([np.random.choice(np.argwhere(digitized == b_t).flatten(), 1) for b_t in neg_batch_bins])
    anchor_pos_neg = np.concatenate((anchor, pos, neg), axis=-1)

    sample_states = normalized_states[anchor_pos_neg.flatten()].reshape(anchor_pos_neg.shape + (state_dim, ))
    sample_actions = actions[anchor_pos_neg.flatten()].reshape(anchor_pos_neg.shape + (action_dim, ))

    torch_states = torch.from_numpy(sample_states).to(dtype=torch.float32, device=device)
    torch_actions = torch.from_numpy(sample_actions).to(dtype=torch.float32, device=device)
    labels = torch.from_numpy(labels).to(dtype=torch.int, device=device)

    return torch_states, torch_actions, labels

def get_batch_contrastive_simclr(data_class: DataClassContrastive):
    batch_size = data_class.batch_size
    num_trajectories = len(data_class.traj_lens)

    batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
    )

    max_ep_len = data_class.max_ep_len
    max_len = data_class.K
    scale = data_class.scale
    device = data_class.device
    state_dim = data_class.state_dim
    action_dim = data_class.action_dim

    normalized_states = data_class.normalized_states
    actions = data_class.actions
    g_t = data_class.rtg
    glob_start_end_indeces = data_class.glob_start_end_indeces
    num_samples_simclr = data_class.num_samples_simclr

    s, a, rtg, = [], [], []
    for i in range(batch_size):
        traj_ix = int(batch_inds[i])
        glob_start_ix, glob_end_ix = glob_start_end_indeces[traj_ix]
        si = random.randint(glob_start_ix, glob_end_ix - num_samples_simclr)
        end_ix = min(si+max_len, glob_end_ix + 1)
        tlen = end_ix - si

        states_per_traj = normalized_states[si:end_ix].reshape(1, -1, state_dim)
        actions_per_traj = actions[si:end_ix].reshape(1, -1, action_dim)
        rtg_per_traj = g_t[si:end_ix].reshape(1, -1, 1)

        s.append(np.concatenate([np.zeros((1, max_len - tlen, state_dim)), states_per_traj], axis=1))
        a.append(np.concatenate([np.ones((1, max_len - tlen, action_dim)) * -10., actions_per_traj], axis=1))
        rtg.append(np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg_per_traj], axis=1) / scale)

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)

    return s, a, rtg

def rvs(dim, loc, scale, size=1, random_state=None):
    if random_state is None:
        random_state = np.random.mtrand._rand
    elif isinstance(random_state, np.integer):
        random_state = np.random.RandomState(random_state)
    else:
        raise Exception('error here')

    size = int(size)
    if size > 1:
        return np.array([rvs(dim, loc, scale, size=1, random_state=random_state)
                         for i in range(size)])

    H = np.eye(dim)
    for n in range(dim):
        x = random_state.normal(loc=loc, scale=scale, size=(dim-n,))
        norm2 = np.dot(x, x)
        x0 = x[0].item()
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0]) if x[0] != 0 else 1
        x[0] += D * np.sqrt(norm2)
        x /= np.sqrt((norm2 - x0**2 + x[0]**2) / 2.)
        # Householder transformation
        H[:, n:] = -D * (H[:, n:] - np.outer(np.dot(H[:, n:], x), x))
    return H
