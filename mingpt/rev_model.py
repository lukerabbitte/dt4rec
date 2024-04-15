"""
GPT model based on MinGPT with new embeddings and sampling
"""

import math
import logging
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from mingpt.rev_utils import read_data

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.att_counter = 0

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Save the att tensor to a new file every 1000 increments
        if self.att_counter % 5000 == 0:
            torch.save(att, os.path.join('/home/luke/code/minGPTrecommendations/attention_dumps/126/', f'att_{self.att_counter}.pt'))

        # Increment the counter
        self.att_counter += 1      

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.proj(y))

        # print(f"what shape is y here? {y.shape}")

        # return y, att  # also return attention weights for visualisation
        # print(f"attention looks like:\n{att.shape}")

        return y  # also return attention weights for visualisation

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_type = config.model_type

        print(f"model_type in gpt module was: {self.model_type}")

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        # print(self.pos_emb.shape) # ([1, 91, 128])
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        # print(self.global_pos_emb.shape) # ([1, 104, 128])
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        # state, action, return_to_go embeddings
        self.state_embedding = nn.Sequential(nn.Linear(4, config.n_embd), nn.Tanh())

        # print(f"self.state_embedding:\n{self.state_embedding}\n")

        if self.model_type == 'reward_only':
            self.reward_embedding = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        elif self.model_type == 'return_to_go_conditioned':
            self.return_to_go_embedding = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        # print(f"self.return_to_go_embedding:\n{self.return_to_go_embedding}\n")
        self.action_embedding = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        # print(f"self.action_embedding:\n{self.action_embedding}\n")

        nn.init.normal_(self.action_embedding[0].weight, mean=0.0, std=0.02)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    """
    self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
    self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size+1, config.n_embd)) # ([1, 91, 128]) 
    self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd)) # ([1, 104, 128])
    self.state_embedding = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
    self.return_to_go_embedding = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
    self.action_embedding = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
    """

    def forward(self, states, actions, targets=None, r=None, timesteps=None):
        # print(f"states in forward is {states.shape}")
        # b, t, s = states.size()   # b is batch size, t is block size, s is unsqueezed dimension
        # assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        # print(f"\nstates of size: {states.size()} looks like:\n {states}\n")
        # print(f"\ntargets of size: {targets.size()} looks like:\n {targets}\n")

        # state before embedding was size ([64, 30, 1]), state after should be ([64, 30, 128])
        # print(f"state before embedding: {states}")

        state_embeddings = self.state_embedding(states.type(torch.float32).contiguous())  # (batch, n_embd)
        # print(f"state_embeddings: {state_embeddings.shape}")

        if actions is not None:
            if self.model_type == 'reward_only':
                reward_embeddings = self.reward_embedding(r.type(torch.float32))
            elif self.model_type == 'return_to_go_conditioned':
                return_to_go_embeddings = self.return_to_go_embedding(r.type(torch.float32))

            action_embeddings = self.action_embedding(
                actions.type(torch.long).squeeze(-1)
            )  # (batch, block_size, n_embd)
            # actions will only have context of (context_length - 1) if there are no targets given
            # print(f"action_embeddings before of size: {action_embeddings.shape} and looks like: action_embeddings")
            action_embeddings = action_embeddings[:, -states.shape[1] + int(targets is None):, :]
            # print(f"action_embeddings after of size: {action_embeddings.shape} and looks like: action_embeddings")

            # print(f"return_to_go_embeddings shape: {return_to_go_embeddings.shape}")
            # print(f"action_embeddings shape: {action_embeddings.shape}")

            middle_shape = states.shape[1] * 3 - int(targets is None)

            # Create the shape of our threefold token embeddings
            token_embeddings = torch.zeros(
                (states.shape[0], middle_shape, self.config.n_embd),
                dtype=torch.float32,
                device=state_embeddings.device
            )

            if self.model_type == 'reward_only':
                token_embeddings[:, ::3, :] = reward_embeddings
            elif self.model_type == 'return_to_go_conditioned':
                token_embeddings[:, ::3, :] = return_to_go_embeddings
            token_embeddings[:, 1::3, :] = state_embeddings
            token_embeddings[:, 2::3, :] = action_embeddings


        elif actions is None:  # only happens at very first timestep of evaluation
            if self.model_type == 'reward_only':
                reward_embeddings = self.reward_embedding(r.type(torch.float32))
            elif self.model_type == 'return_to_go_conditioned':
                return_to_go_embeddings = self.return_to_go_embedding(r.type(torch.float32))
            middle_shape = states.shape[1] * 2
            token_embeddings = torch.zeros((states.shape[0], middle_shape, self.config.n_embd),
                                           dtype=torch.float32, device=state_embeddings.device)

            if self.model_type == 'reward_only':
                token_embeddings[:, ::2, :] = reward_embeddings
            elif self.model_type == 'return_to_go_conditioned':
                token_embeddings[:, ::2, :] = return_to_go_embeddings  # really just [:,0,:]
            token_embeddings[:, 1::2, :] = state_embeddings  # really just [:,1,:]

        # do position embeddings - global_pos_emb of size ([1, max(timestep)+1, n_embd]) or ([1, 104, 128])
        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0)

        position_embeddings = torch.gather(
            all_global_pos_emb,
            1,
            torch.repeat_interleave(
                timesteps,
                self.config.n_embd,
                dim=-1
            )
        ) + self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)  # all size (64, 90, 128)
        logits = self.head(x)  # now becomes size (64, 90, 273)

        if actions is not None:
            # The prediction head corresponding to the input token 'st' is trained to predict 'at'
            action_logits = logits[:, 2::3, :]
            logits = logits[:, 1::3, :]
            # print(f"logits shape after resize:\n{logits.shape}\n") ([128, 30, 273])
        elif actions is None:
            logits = logits[:, 1:, :]

        # if we are given some desired targets also calculate the loss
        loss = None
        action_loss = None

        # An entire batch of raw logits can be fed into cross entropy function
        if targets is not None:
            # Make it easier to understand dims for cross-entropy
            input = logits.permute(0, 2, 1)  # ([64, 273, 30])
            action_input = logits.permute(0, 2, 1)  # ([64, 273, 30])
            target = targets.view(targets.size(0), -1)  # ([64, 30])
            loss = F.cross_entropy(input, target)
            action_loss = F.cross_entropy(action_input, target)


            """
            # Rough but can I compare to complete matrix here? Just to give idea
            eval_states, eval_actions, eval_rewards, _, _, eval_timesteps, eval_terminal_indices = read_data(
                'data/goodreads_eval_modified.tsv')
            print("user_id is following")
            curr_user_id = int(states[-1][-1].squeeze(0).item())
            print(curr_user_id)
            

            # Before, we got the prediction based on final timestep - logits[:, -1, :]
            # Now, we just want final batch, for *each* timestep, not just for the last - logits[-1, :, :]
            logits_sample = logits[-1, :, :]
            probs = F.softmax(logits_sample, dim=-1)
            print(probs.shape)
            ix = torch.multinomial(probs, num_samples=1)
            print(ix.shape) # 64, 30
            print(ix)
            ix_list = ix.squeeze().tolist()
            ix_list = [x + 1 for x in ix_list]
            print(ix_list)


            # Now get the 30 targets, one for each timestep.
            fine_input = input[-1]
            print(f"target shape: {target.shape}")
            fine_target = target[-1]
            print(fine_input.shape)
            print(fine_target.squeeze(0))
            """


        # print(f"ix is {ix}")
        # for i in range(states.size(0)):
        #     print(f"prediction for user {states[i][1].squeeze(0)} is {ix[i].squeeze(0) + 1}")

        return logits, loss, action_loss