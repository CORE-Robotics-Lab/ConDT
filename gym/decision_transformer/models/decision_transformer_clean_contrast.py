import torch
import torch.nn as nn

import transformers

from decision_transformer.models.model import TrajectoryModel
from decision_transformer.models.trajectory_gpt2 import GPT2Model
from decision_transformer.models.decision_transformer_clean import DecisionTransformer_Product

class DecisionTransformerContrast(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            compress_dim,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.compress_dim = compress_dim

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.compress = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.compress_dim),
            nn.ReLU(inplace=True)
        )

        self.discriminator = nn.Linear(2*self.compress_dim, 1)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward_contrast(self, states, actions, rtg):
        """
        Run Contrastive Learning

        :param states: batch x 3 x state_dim
        :param actions: batch x 3 x action_dim
        :return: (batch x 1, batch x 1)
        """
        latents = self.get_latent(states, actions, rtg)
        anchor = latents[:, 0]
        pos = latents[:, 1]
        neg = latents[:, 2]

        return self.discriminator(torch.cat((anchor, pos), dim=-1)), self.discriminator(torch.cat((anchor, neg), dim=-1))

    def get_latent(self, states, actions, rtg):
        """
        Get latent embedding of (s, a) pair

        :param states: batch x 3 x state_dim
        :param actions: batch x 3 x action_dim
        :return: (batch x 3 x self.compress_dim)
        """
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        s_a_embeddings = torch.cat([state_embeddings, action_embeddings], dim=-1)
        compress = self.compress(s_a_embeddings)
        return compress

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack( #this is [batch_size x 3 x seq_len x dim] 3 -> (R, s, a)
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # predict next return given state and action
        state_preds = self.predict_state(x[:,2])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(
            states, actions, returns_to_go, timesteps, attention_mask=attention_mask)

        return action_preds[0,-1]

    def get_input_params(self, w_names=False):
        inputs = ['embed_state', 'embed_return', 'embed_action', 'compress', 'discriminator']
        valid_params = []
        for name, param in self.named_parameters():
            for valid_input in inputs:
                if valid_input in name:
                    valid_params.append((name, param) if w_names else param)
                    break
        if w_names:
            return dict(valid_params)
        else:
            return valid_params

    def get_non_input_params(self, w_names=False):
        inputs = ['embed_state', 'embed_return', 'embed_action', 'compress', 'discriminator']
        valid_params = []
        for name, param in self.named_parameters():
            not_in = True
            for valid_input in inputs:
                if valid_input in name:
                    not_in = False
                    break

            if not_in:
                valid_params.append((name, param) if w_names else param)
        if w_names:
            return dict(valid_params)
        else:
            return valid_params

"""
This is the same as the above, except we use the latent (s,a) embeddings trained by the contrastive model as
input to the DT
"""
class DecisionTransformerContrast_UseLatent(DecisionTransformerContrast):
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            compress_dim,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super(DecisionTransformerContrast_UseLatent, self).__init__(state_dim, act_dim, hidden_size, compress_dim,
                                                                    max_length, max_ep_len, action_tanh, **kwargs)
        self.compress_dim = compress_dim

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=self.compress_dim,
            **kwargs
        )

        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, self.compress_dim)

        self.embed_return = torch.nn.Linear(1, self.compress_dim)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.compress = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.compress_dim),
            nn.ReLU(inplace=True)
        )

        self.discriminator = nn.Sequential(
            nn.Linear(2*self.compress_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        self.embed_ln = nn.LayerNorm(self.compress_dim)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(self.compress_dim, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.compress_dim, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(self.compress_dim, 1)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states) # batch x seq_length x hidden_size
        action_embeddings = self.embed_action(actions) # batch x seq_length x hidden_size
        returns_embeddings = self.embed_return(returns_to_go) # batch x seq_length x hidden_size
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        s_a_embeddings = self.compress(torch.cat([state_embeddings, action_embeddings], dim=-1)) + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack( #this is [batch_size x 2 x seq_len x compress_dim] 2 -> (s_a, rtg)
            (returns_embeddings, s_a_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.compress_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), s_a (1)
        x = x.reshape(batch_size, seq_length, 2, self.compress_dim).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:, 0])  # predict next return given state and action
        state_preds = self.predict_state(x[:,1])    # predict next state given state and action
        action_preds = self.predict_action(x[:,1])  # predict next action given state

        return state_preds, action_preds, return_preds


class DecisionTransformerContrast_Product(DecisionTransformer_Product):
    def __init__(self, state_dim, act_dim, hidden_size, compress_dim, max_length=None, max_ep_len=4096,
            action_tanh=True, **kwargs):
        super(DecisionTransformerContrast_Product, self).__init__(state_dim, act_dim, hidden_size, max_length, max_ep_len,
                                                                  action_tanh, **kwargs)
        self.compress_dim = compress_dim
        self.compress = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.compress_dim),
            nn.ReLU(inplace=True)
        )
        self.discriminator = nn.Linear(2*self.compress_dim, 1)

    def get_latent(self, states, actions, rtg):
        # states, actions, rtg can either be: [batch|_|state_dim, batch|_|action_dim, batch|_|1] or [batchxstate_dim, batchxaction_dim, batchx1]
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(rtg.unsqueeze(-1))

        state_embeddings = state_embeddings*returns_embeddings
        action_embeddings = action_embeddings*returns_embeddings

        s_a_embeddings = torch.cat([state_embeddings, action_embeddings], dim=-1)
        compress = self.compress(s_a_embeddings)
        return compress

    def forward_contrast(self, states, actions, rtg):
        """
        Run Contrastive Learning

        :param states: batch x 3 x state_dim
        :param actions: batch x 3 x action_dim
        :return: (batch x 1, batch x 1)
        """
        latents = self.get_latent(states, actions, rtg)
        anchor = latents[:, 0]
        pos = latents[:, 1]
        neg = latents[:, 2]

        return self.discriminator(torch.cat((anchor, pos), dim=-1)), self.discriminator(torch.cat((anchor, neg), dim=-1))

class DecisionTransformerContrast_SIMCLR(DecisionTransformerContrast):
    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            compress_dim,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super(DecisionTransformerContrast_SIMCLR, self).__init__(
            state_dim, act_dim, hidden_size, compress_dim, max_length, max_ep_len,
            action_tanh, **kwargs)
        self.embed_state = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, compress_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.embed_action = nn.Sequential(
            nn.Linear(act_dim, hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, compress_dim),
            nn.LeakyReLU(inplace=True)
        )

        self.compress = nn.Linear(2*compress_dim, compress_dim)

class DecisionTransformerContrast_ProductSIMCLR(DecisionTransformer_Product):
    def __init__(self, state_dim, act_dim, hidden_size, compress_dim, max_length=None, max_ep_len=4096,
                 action_tanh=True, **kwargs):
        super(DecisionTransformerContrast_ProductSIMCLR, self).__init__(state_dim, act_dim, hidden_size, max_length,
                                                                        max_ep_len, action_tanh, **kwargs)
        self.embed_state = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, compress_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.embed_action = nn.Sequential(
            nn.Linear(act_dim, hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_size, compress_dim),
            nn.LeakyReLU(inplace=True)
        )

        self.compress = nn.Linear(2*compress_dim, compress_dim)

    def get_latent(self, states, actions, rtg):
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(rtg)

        state_embeddings = state_embeddings*returns_embeddings
        action_embeddings = action_embeddings*returns_embeddings

        s_a_embeddings = torch.cat([state_embeddings, action_embeddings], dim=-1)
        compress = self.compress(s_a_embeddings)
        return compress
