"""
The main driver program for running experiments on the model
"""

import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from mingpt.utils import set_seed
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as Fun
from torch.utils.data import Dataset
from mingpt.rev_model import GPT, GPTConfig
from mingpt.rev_trainer import Trainer, TrainerConfig
from mingpt.rev_utils import read_data
import argparse
from mingpt.rev_utils import plot_loss
from mingpt.rev_utils import plot_reward

seed = 123
epochs = 150
batch_size = 256
context_length = 1
exp_number = 141

train_dataset_filename = 'data/goodreads_data_first_2048_users_4_groups_beta_distribution.tsv'
test_dataset_filename = 'data/dummy_50.tsv'
eval_dataset_filename = 'data/goodreads_eval.tsv'
eval_data_filename = 'data/goodreads_eval_first_50.tsv'

model_type = 'return_to_go_conditioned'  # reward_only || return_to_go_conditioned

class ReviewDataset(Dataset):

    def __init__(self, states, actions, rewards, returns, returns_to_go, timesteps, terminal_indices, block_size, model_type):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.returns = returns
        self.returns_to_go = returns_to_go
        self.timesteps = timesteps
        self.terminal_indices = terminal_indices
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.model_type = model_type

        # print(f"Previously, states were all different user ids {self.states}")
        # Change states at this stage to make constant - all 0
        self.states = [0] * len(self.states)

    def get_max_return(self):
        # Highest return from training data
        max_return = max(self.returns)
        print(f"max individual return: {max_return}")
        return max_return

    def __len__(self):
        return len(self.states) - self.block_size

    def __getitem__(self, idx):
        # print(f"original idx given was {idx}")
        block_size = self.block_size // 3   # aka, the original context length
        done_idx = idx + block_size
        for i in self.terminal_indices:
            if i > idx:  # find the first terminal index greater than idx
                done_idx = min(i, done_idx)
                break

        idx = done_idx - block_size

        # Squeeze these tensors to give dimension for batch size expected by most APIs (b,t)
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.float32).unsqueeze(1)
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.long)
        states = Fun.one_hot(states, num_classes=4)
        states = states.float()

        # print(f"states is really like: {states.shape}")
        # states = states.unsqueeze(-1)

        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # was (block_size, 1) back when there was an unsqueeze
        returns_to_go = torch.tensor(self.returns_to_go[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(self.rewards[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        # print(f"states.size: {states.shape}")
        # print(f"actions.size: {actions.shape}")
        # print(f"rewards.size: {returns_to_go}")
        # print(f"timesteps.size: {timesteps.shape}")

        if model_type == 'reward_only':
            return states, actions, rewards, timesteps # TODO rtgs or rewards?
        elif model_type == 'return_to_go_conditioned':
            return states, actions, returns_to_go, timesteps  # TODO rtgs or rewards?


class EvalDataset(Dataset):

    def __init__(self, states, actions, rewards, timesteps, terminal_indices, block_size):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.timesteps = timesteps
        self.terminal_indices = terminal_indices
        self.block_size = block_size
        self.vocab_size = max(actions) + 1

        # Get indices denoting start of each user's interaction trajectory
        self.start_indices = self.terminal_indices
        self.start_indices = np.insert(self.start_indices, 0, 0)

    def __len__(self):
        return len(self.states)

    # Returns user data from a complete matrix of user interactions where they have rated every item, for eval purposes
    # Also note that each of our terminal indices will be one index higher than the last entry for a user.
    # This is in keeping with Python's upper bound exclusive behaviour.
    def __getitem__(self, user_id):
        # print(f"user_id passed to EvalDataset getitem is: {user_id}")
        idx = self.start_indices[user_id - 1]
        done_idx = None if user_id == self.start_indices.size else self.terminal_indices[
            user_id - 1]  # avoid array out of limit bug for last user

        # Return tensors of (episode_length, 1)
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(self.rewards[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rewards, timesteps

# Read in train data and create dataset
train_states, train_actions, train_rewards, train_returns, train_returns_to_go, train_timesteps, train_terminal_indices = read_data(
    train_dataset_filename)
train_dataset = ReviewDataset(train_states, train_actions, train_rewards, train_returns, train_returns_to_go, train_timesteps, train_terminal_indices, context_length * 3, model_type)
len_train_dataset = len(train_states)
max_training_return = train_dataset.get_max_return()


# Read in test data and create dataset
test_states, test_actions, test_rewards, test_returns, test_returns_to_go, test_timesteps, test_terminal_indices = read_data(test_dataset_filename)
test_dataset = ReviewDataset(test_states, test_actions, test_rewards, test_returns_to_go, test_timesteps, test_timesteps, test_terminal_indices, context_length * 3, model_type)
len_test_dataset = len(test_states)

eval_states, eval_actions, eval_rewards, _, _, eval_timesteps, eval_terminal_indices = read_data(
    eval_dataset_filename)
eval_dataset = EvalDataset(eval_states, eval_actions, eval_rewards, eval_timesteps, eval_terminal_indices, context_length * 3)
len_eval_dataset = len(eval_states)

# This is simple dataframe for taking average across 50 random users from group 1
eval_data = pd.read_csv(eval_data_filename, sep='\t')

# print(f"max_timesteps across entire dataset is: {max(timesteps)}")
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=6, n_head=8,
                  n_embd=128, model_type=model_type, max_timestep=max(train_timesteps))
model = GPT(mconf)

print(train_dataset.vocab_size, train_dataset.block_size, max(train_timesteps))

# initialize a trainer instance and kick off tra ning
tconf = TrainerConfig(max_epochs=epochs, batch_size=batch_size, learning_rate=0.0001,
                      lr_decay=False, warmup_tokens=512 * 20,
                      final_tokens=2 * len(train_dataset) * context_length * 3,
                      num_workers=4, seed=seed, model_type=model_type,
                      ckpt_dir=f"new_checkpoints/{exp_number}",
                      max_timestep=max(train_timesteps),
                      num_users=2048,
                      ratings_per_user=None,
                      num_recs=10,
                      ratings_at_extreme=False,
                      exp_number=exp_number,
                      max_return=max_training_return)

trainer = Trainer(model, train_dataset, None, tconf, eval_dataset, eval_data)
train_losses, action_losses, test_losses, rewards_per_epoch = trainer.train()

plot_loss(train_losses, test_losses, context_length, batch_size, mconf.n_layer, mconf.n_head, mconf.n_embd, train_dataset_filename, len_train_dataset, test_dataset_filename, len_test_dataset,
          tconf.learning_rate, tconf.lr_decay, tconf.num_users, tconf.ratings_at_extreme, model_type, exp_number)


plot_reward(rewards_per_epoch, context_length, batch_size, mconf.n_layer, mconf.n_head, mconf.n_embd,
              train_dataset_filename, len_train_dataset, tconf.learning_rate, tconf.lr_decay, tconf.num_users, tconf.num_recs, tconf.ratings_at_extreme, model_type, exp_number)

print(f"train_losses: {train_losses}")
print(f"test_losses: {test_losses}")
print(f"rewards_per_epoch: {rewards_per_epoch}")
print(context_length, batch_size, mconf.n_layer, mconf.n_head, mconf.n_embd,
              train_dataset_filename, len_train_dataset, tconf.learning_rate, tconf.lr_decay, tconf.num_users, tconf.num_recs, model_type)

with open(f"experiments/experiment{exp_number}.txt", "w") as file:
    # Write the variables to the file
    file.write(f"train_losses: {train_losses}\n")
    file.write(f"rewards_per_epoch: {rewards_per_epoch}\n")
    file.write(f"context_length: {context_length}\n")
    file.write(f"batch_size: {batch_size}\n")
    file.write(f"mconf.n_layer: {mconf.n_layer}\n")
    file.write(f"mconf.n_head: {mconf.n_head}\n")
    file.write(f"mconf.n_embd: {mconf.n_embd}\n")
    file.write(f"train_dataset_filename: {train_dataset_filename}\n")
    file.write(f"len_train_dataset: {len_train_dataset}\n")
    file.write(f"tconf.learning_rate: {tconf.learning_rate}\n")
    file.write(f"tconf.lr_decay: {tconf.lr_decay}\n")
    file.write(f"tconf.num_users: {tconf.num_users}\n")
    file.write(f"tconf.num_recs: {tconf.num_recs}\n")
    file.write(f"model_type: {model_type}\n")