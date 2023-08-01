import numpy as np
import torch
import wandb
from pytorch_metric_learning.losses import NTXentLoss

from decision_transformer.training.trainer_clean import BaseTrainer
from decision_transformer.utils.data import anchor_inds
from decision_transformer.models.decision_transformer_clean import DecisionTransformer_Rotation

class SequenceTrainer(BaseTrainer):
    def __init__(self, model, optimizer, batch_size, get_batch, data_class, loss_fn, env, scheduler=None, eval_fn=None,
                 log_to_wandb=True):
        super(SequenceTrainer, self).__init__(model, optimizer, batch_size, get_batch, data_class,
                                              loss_fn, env, scheduler, eval_fn, log_to_wandb)

    def train_step(self, iter_ix):
        states, actions, rtg, timesteps, attention_mask, path_inds = self.get_batch(self.data_class)
        action_target = torch.clone(actions)

        if isinstance(self.model, DecisionTransformer_Rotation):
            digitized = torch.from_numpy(self.data_class.digitized).to(device=path_inds.device, dtype=torch.long)
            bin_index = torch.zeros_like(path_inds)
            bin_index[path_inds != -1] = digitized[path_inds[path_inds != -1]]
            bin_index[path_inds == -1] = 1
            bin_index = bin_index.clamp(min=1, max=self.data_class.num_bins)

            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rtg, timesteps, bin_index, attention_mask=attention_mask,
            )
        else:
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rtg, timesteps, attention_mask=attention_mask,
            )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return {'train_loss': loss.detach().cpu().item()}

class SequenceTrainerContrastive(BaseTrainer):
    def __init__(self, model, optimizer, batch_size, get_batch, get_batch_contrastive, data_class, loss_fn, env,
                 scheduler=None, eval_fn=None, log_to_wandb=True, use_same_inds=False, contrastive_frequency = 1):
        super(SequenceTrainerContrastive, self).__init__(model, optimizer, batch_size, get_batch, data_class,
                                                         loss_fn, env, scheduler, eval_fn, log_to_wandb)
        self.get_batch_contrastive = get_batch_contrastive
        self.rcrl_loss = torch.nn.MSELoss(reduction='sum')
        self.use_same_inds = use_same_inds
        self.contrastive_frequency = contrastive_frequency

    def train_step(self, iter_ix):
        inds = None
        if self.use_same_inds:
            inds = anchor_inds(self.data_class)

        states, actions, rtg, timesteps, attention_mask, _ = self.get_batch(self.data_class, inds)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rtg, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        sample_state, sample_actions, sample_rtg, _ = self.get_batch_contrastive(self.data_class, inds)

        pos_diff, neg_diff = self.model.forward_contrast(sample_state, sample_actions, sample_rtg)
        pos_target, neg_diff = torch.ones_like(pos_diff), torch.zeros_like(neg_diff)

        contrastive_loss = self.rcrl_loss(pos_diff, pos_target) + self.rcrl_loss(neg_diff, neg_diff)
        contrastive_loss /= 2*self.batch_size

class SequenceTrainerContrastive_SIMCLR(BaseTrainer):
    def __init__(self, model, optimizer, batch_size, get_batch, get_batch_contrastive, data_class, loss_fn, env,
                 scheduler=None, eval_fn=None, log_to_wandb=True, contrastive_frequency = 1, beta=0.1):
        super(SequenceTrainerContrastive_SIMCLR, self).__init__(model, optimizer, batch_size, get_batch, data_class,
                                                         loss_fn, env, scheduler, eval_fn, log_to_wandb)
        self.get_batch_contrastive = get_batch_contrastive
        self.contrastive_frequency = contrastive_frequency
        self.using_pretrain=False
        self.beta = beta

    def pretrain(self):
        self.using_pretrain = True
        contrastive_loss = self.simclr_loss()

        self.optimizer.zero_grad()
        contrastive_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        return {'train_loss': torch.zeros_like(contrastive_loss).detach().cpu().item(), 'contrastive_loss': contrastive_loss.detach().cpu().item()}

    def simclr_loss(self):
        sample_state, sample_actions, sample_rtg = self.get_batch_contrastive(self.data_class)

        B, N, state_dim = sample_state.shape
        _, _, action_dim = sample_actions.shape

        num_positives = self.data_class.num_samples_simclr
        ixs = np.concatenate([np.random.choice(np.arange(N), size=num_positives).reshape(1, -1) for _ in range(B)], axis=0)
        torch_ixs = torch.from_numpy(ixs).to(device=self.data_class.device)
        pos_states = torch.cat([sample_state[b_ix, torch_ixs[b_ix]].unsqueeze(0) for b_ix in range(B)])
        pos_actions = torch.cat([sample_actions[b_ix, torch_ixs[b_ix]].unsqueeze(0) for b_ix in range(B)])
        pos_rtg = torch.cat([sample_rtg[b_ix, torch_ixs[b_ix]].unsqueeze(0) for b_ix in range(B)])

        pos_latents = self.model.get_latent(pos_states, pos_actions, pos_rtg)
        proj_out_dim = pos_latents.shape[-1]

        labels = torch.arange(0, B).reshape(-1, 1).repeat(1, num_positives).flatten().to(self.data_class.device)
        simclr_loss = NTXentLoss(temperature=0.1)

        loss = simclr_loss(embeddings = pos_latents.reshape(-1, proj_out_dim), labels=labels)
        return loss


    def train_step(self, iter_ix):
        states, actions, rtg, timesteps, attention_mask, _ = self.get_batch(self.data_class)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rtg, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        contrastive_loss = self.simclr_loss()

        self.optimizer.zero_grad()
        if iter_ix % self.contrastive_frequency == 0:
            (loss + self.beta*contrastive_loss).backward()
        else:
            (loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return {'train_loss': loss.detach().cpu().item(), 'contrastive_loss': contrastive_loss.detach().cpu().item()}


