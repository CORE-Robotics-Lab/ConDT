import itertools

import numpy as np
import wandb
import time
import pandas as pd


class BaseTrainer:

    def __init__(self, model, optimizer, batch_size, get_batch, data_class, loss_fn, env, scheduler=None, eval_fn=None,
                 log_to_wandb=True):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.data_class = data_class
        self.loss_fn = loss_fn
        self.env = env
        self.scheduler = scheduler
        self.eval_fn = eval_fn
        self.diagnostics = dict()
        self.log_to_wandb = log_to_wandb
        self.start_time = time.time()

        self.target_rew = self.data_class.env_targets[0]
        data_columns = list(itertools.chain.from_iterable([
            [f'seed_{seed}_target_{self.target_rew}_return_mean',
             f'seed_{seed}_target_{self.target_rew}_return_std',
             f'seed_{seed}_target_{self.target_rew}_length_mean',
             f'seed_{seed}_target_{self.target_rew}_length_std'
             ] for seed in [1, 5, 10]]))

        self.eval_table = pd.DataFrame(columns=data_columns + ['return_mean', 'return_std'])

    def eval_method(self, logs={}):
        self.model.eval()
        seeded_outputs = self.eval_fn(self.model, self.data_class, self.env)
        output_info = {}
        json_seeded_outputs = {}

        mean_seeded_return = []
        mean_seeded_length = []
        for seed, outputs in seeded_outputs.items():
            mean_return = np.mean(outputs['returns'])
            mean_length = np.mean(outputs['lengths'])

            mean_seeded_return.append(mean_return)
            mean_seeded_length.append(mean_length)

            output_info[f'seed_{seed}_target_{self.target_rew}_return_mean'] = mean_return
            output_info[f'seed_{seed}_target_{self.target_rew}_return_std'] = np.std(outputs['returns'])
            output_info[f'seed_{seed}_target_{self.target_rew}_length_mean'] = mean_length
            output_info[f'seed_{seed}_target_{self.target_rew}_length_std'] = np.std(outputs['lengths'])

        output_info['return_mean'] = np.mean(mean_seeded_return)
        output_info['return_std'] = np.std(mean_seeded_return)

        for k,v in output_info.items():
            logs[f'evaluation/{k}'] = v

        self.eval_table.loc[len(self.eval_table)] = output_info
        if self.log_to_wandb:
            wandb.log({"eval_table": wandb.Table(columns=list(self.eval_table.columns),
                                                 data=self.eval_table.values.tolist())})
            wandb.log({
                'mean_seed_return': np.mean(mean_seeded_return),
                'mean_seed_length': np.mean(mean_seeded_length)
            })

        return seeded_outputs

    def train_iteration(self, num_steps, iter_num=0, print_logs=True):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for ix in range(num_steps):
            if iter_num == 0:
                train_loss = self.pretrain()
            else:
                train_loss = self.train_step(ix + (iter_num - 1)*num_steps)
            train_losses.append(train_loss)
            if self.log_to_wandb:
                wandb.log(train_loss)
            if self.scheduler is not None and iter_num != 0:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()
        self.model.eval()
        json_outputs = self.eval_method(logs)
        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        df = pd.DataFrame(train_losses)
        mean = df.mean(axis=0)
        std = df.std(axis=0)
        for col in df.columns:
            logs[f'{col}_mean'] = mean[col]
            logs[f'{col}_std'] = std[col]

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            self.print_logs(logs, iter_num)

        return json_outputs

    def print_logs(self, logs, iter_num):
        print('=' * 80)
        print(f'Iteration {iter_num}')
        for k, v in logs.items():
            print(f'{k}: {v}')
        return logs

    def pretrain(self):
        raise NotImplementedError

    def train_step(self, iter_ix):
        raise NotImplementedError()
