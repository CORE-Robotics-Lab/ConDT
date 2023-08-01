"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os

import pandas as pd
import wandb
from pytorch_metric_learning.losses import NTXentLoss
import platform
from mingpt.model_atari import GPT, GPT_Rotation

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image
from mingpt.utils import get_bin_ranges, get_digitized

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 1 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            print('USING CUDA!')
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            if isinstance(model, GPT_Rotation):
                model.rotation = model.rotation.to(device=self.device)
            #self.model = torch.nn.DataParallel(self.model).to(self.device)

        if config.log_to_wandb:
            self.log_to_wandb = True
            env_name = self.config.game
            name_experiment = self.config.name
            exp_prefix = 'atari-contrast-exp_2'
            group_name = f'{exp_prefix}-{env_name}'
            exp_prefix = f'{group_name}-{name_experiment}{"-scaled" if self.config.scaled else ""}{"-pretrain" if self.config.pretrain else ""}'

            wandb.init(
                name=exp_prefix,
                group=group_name,
                project='decision-transformer',
                config=self.config
            )
        else:
            self.log_to_wandb = False

        self.target_rew = self.config.target
        data_columns = [f'target_{self.target_rew}_return_mean', f'target_{self.target_rew}_return_std']
        self.eval_table = pd.DataFrame(columns=data_columns)

        self.bin_ranges = self.config.bin_ranges

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def get_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            rtg: torch.Tensor,
            use_simclr=True
    ):
        B, N, state_dim = states.shape
        _, _, action_dim = actions.shape

        num_positives = 2 if self.config.game != 'Pong' else 3
        ixs = np.concatenate([np.random.choice(np.arange(N), size=num_positives).reshape(1, -1) for _ in range(B)], axis=0)
        torch_ixs = torch.from_numpy(ixs).to(device=self.device)
        if use_simclr:
            pos_states = torch.cat([states[b_ix, torch_ixs[b_ix]].unsqueeze(0) for b_ix in range(B)])
            pos_actions = torch.cat([actions[b_ix, torch_ixs[b_ix]].unsqueeze(0) for b_ix in range(B)])
            pos_rtg = torch.cat([rtg[b_ix, torch_ixs[b_ix]].unsqueeze(0) for b_ix in range(B)])

            pos_latents = self.model.get_latent(pos_states, pos_actions, pos_rtg)
            proj_out_dim = pos_latents.shape[-1]

            labels = torch.arange(0, B).reshape(-1, 1).repeat(1, num_positives).flatten().to(self.device)
            simclr_loss = NTXentLoss(temperature=0.1)
            loss = simclr_loss(embeddings = pos_latents.reshape(-1, proj_out_dim), labels=labels)
            return loss
        else:
            raise Exception('hasnt been handled yet')

    def train(self, contrastive=False, pretrain=False):
        if not contrastive and pretrain:
            raise Exception('Cant contrastive pretrain with a non-contrastive model!')

        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config, pretrain=pretrain)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            if platform.system() != 'Linux':
                loader = DataLoader(data, shuffle=True, batch_size=config.batch_size)
            else:
                loader = DataLoader(data, shuffle=True, batch_size=config.batch_size, num_workers=2)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, r, t) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    if isinstance(model, GPT_Rotation):
                        logits, loss = model(x, y, y, r/self.config.rtg_scale, t, digitized=get_digitized(r, self.bin_ranges))
                    else:
                        logits, loss = model(x, y, y, r/self.config.rtg_scale, t)

                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus

                    epoch_loss = [loss]
                    if contrastive and not pretrain:
                        if self.config.game == 'Pong':
                            batch_size = 64
                        else:
                            batch_size = 32
                        batch_inds = np.array(random.sample([ix for ix in range(len(self.train_dataset))], k=batch_size))

                        s_s, a_s, rtg_s = [], [], []
                        for ind in batch_inds:
                            s, a, rtg = self.train_dataset.get_contrastive(ind, 15)
                            s_s.append(s.unsqueeze(0))
                            a_s.append(a.unsqueeze(0))
                            rtg_s.append(rtg.unsqueeze(0))
                        s_s = torch.cat(s_s, dim=0).to(self.device)
                        a_s = torch.cat(a_s, dim=0).to(self.device)
                        rtg_s = torch.cat(rtg_s, dim=0).to(self.device)

                        contrastive_loss = self.get_loss(s_s, a_s, rtg_s/self.config.rtg_scale)
                        epoch_loss.append(contrastive_loss)
                    losses.append(loss.item())

                if self.log_to_wandb:
                    keys = ['train_loss', 'contrastive_loss']
                    log_obj = {keys[ix]:epoch_loss[ix].clone().detach().cpu().item() for ix in range(len(epoch_loss))}
                    wandb.log(log_obj)

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    weights = [1, 0.1]
                    sum([epoch_loss[ix]*weights[ix] for ix in range(len(epoch_loss))]).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    if not contrastive or pretrain:
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {epoch_loss[0].item():.5f}. lr {lr:e}")
                    else:
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {epoch_loss[0].item():.5f}. Contrastive loss {epoch_loss[1].item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        # best_loss = float('inf')

        if pretrain and contrastive:
            pbar = tqdm(range(2000))
            for epoch in pbar:
                batch_inds = np.array(random.sample([ix for ix in range(len(self.train_dataset))], k=32))

                s_s, a_s, rtg_s = [], [], []
                for ind in batch_inds:
                    s, a, rtg = self.train_dataset.get_contrastive(ind, 15)
                    s_s.append(s.unsqueeze(0))
                    a_s.append(a.unsqueeze(0))
                    rtg_s.append(rtg.unsqueeze(0))
                s_s = torch.cat(s_s, dim=0).to(self.device)
                a_s = torch.cat(a_s, dim=0).to(self.device)
                rtg_s = torch.cat(rtg_s, dim=0).to(self.device)
                contrastive_loss = self.get_loss(s_s, a_s, rtg_s/self.config.rtg_scale)

                model.zero_grad()
                contrastive_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                scalar_loss = contrastive_loss.clone().detach().cpu().item()
                if self.log_to_wandb:
                    log_obj = {'contrastive_loss':scalar_loss}
                    wandb.log(log_obj)

                pbar.set_description(f"epoch {epoch+1}: contrastive loss {scalar_loss:.5f}.)")

            #print('-Halfing learning rate of encoders!')
            #for param_ix in range(0,2):
            #    optimizer.param_groups[param_ix]['lr'] /= 2

        best_return = -float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            run_epoch('train', epoch_num=epoch)
            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')

            # # supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()

            # -- pass in target returns
            if self.config.model_type == 'naive':
                eval_return = self.get_returns(0)
            elif self.config.model_type == 'reward_conditioned':
                eval_return = self.get_returns(self.target_rew)
                eval_table_ix = {
                    f'target_{self.target_rew}_return_mean': np.mean(eval_return),
                    f'target_{self.target_rew}_return_std': np.std(eval_return),
                }

                self.eval_table.loc[len(self.eval_table)] = eval_table_ix
                if self.log_to_wandb:
                    wandb.log(
                        eval_table_ix
                    )

                    wandb.log({"eval_table": wandb.Table(columns=list(self.eval_table.columns),
                                                         data=self.eval_table.values.tolist())})

            else:
                raise NotImplementedError()

            if self.log_to_wandb:
                torch.save(model.state_dict(), os.path.join(self.config.save_path, f'dt_{epoch}_model.pth'))

    def get_returns(self, ret):
        self.model.train(False)
        args=Args(self.config.game.lower(), self.config.seed)
        env = Env(args)
        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        if isinstance(self.model, GPT_Rotation):
            digi_func = lambda g_t: get_digitized(g_t, self.bin_ranges)
        else:
            digi_func = None
        for i in range(10):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(
                self.model.module if hasattr(self.model, 'module') else self.model, state, 1, temperature=1.0, sample=True, actions=None,
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1)/self.config.rtg_scale,
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device),
                digitized_func= digi_func
            )

            j = 0
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all prexvious rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(
                    self.model.module if hasattr(self.model, 'module') else self.model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1)/self.config.rtg_scale,
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)),
                    digitized_func= digi_func
                )
        env.close()
        eval_return = sum(T_rewards)/10.
        print("target return: %d, eval return: %d" % (ret, eval_return))
        self.model.train(True)
        return np.array(T_rewards)


class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

class Args:
    def __init__(self, game, seed):
        if torch.cuda.is_available():
            print('using CUDA!')
            self.device = torch.device('cuda')
        else:
            print('using CPU!')
            self.device = torch.device('cpu')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
