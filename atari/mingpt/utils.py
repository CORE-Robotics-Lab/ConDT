"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None,
           digitized_func = None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        x_cond = x if x.size(1) <= block_size//3 else x[:, -block_size//3:] # crop context if needed
        if actions is not None:
            actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
        rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:] # crop context if needed

        if digitized_func is not None:
            logits, _ = model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps,
                              digitized=digitized_func(rtgs))
        else:
            logits, _ = model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix

    return x

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

def get_bin_ranges(g_t, num_bins, max_val=None):
    max_g_t = max_val if max_val is not None else g_t.max()
    bins = np.linspace(0, max_g_t + 1, num_bins + 1, dtype=int)
    return bins

def get_digitized(g_t, bin_ranges):
    if isinstance(g_t, torch.Tensor):
        device = g_t.device
        if isinstance(bin_ranges, torch.Tensor):
            bin_ranges = bin_ranges.clone().detach().cpu().numpy()
        g_t = g_t.clone().detach().cpu().numpy()
        digitized = np.digitize(g_t, bin_ranges)
        return torch.from_numpy(digitized).to(dtype=torch.long, device=device)
    else:
        digitized = np.digitize(g_t, bin_ranges)
        return digitized