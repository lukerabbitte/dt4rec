"""
Utility functions used throughout the training, sampling, evaluation, and plotting
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
def sample(model, x, y, r, t, steps, temperature=1.0, sample=False, num_recs=1, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.

    Note model signature is the following:
    def forward(self, states, actions, targets=None, returns_to_go=None, timesteps=None):
    """
    # block_size = model.get_block_size()
    block_size = 3 # TODO hard-coded
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size//3 else x[:, -block_size//3:, :] # crop context if needed
        if y is not None:
            y = y if y.size(1) <= block_size//3 else y[:, -block_size//3:, :]
        r = r if r.size(1) <= block_size//3 else r[:, -block_size//3:, :]
        logits, _, _ = model(x_cond, actions=y, targets=None, r=r, timesteps=t)

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # logits = logits / temperature
        # print(f"logits shape is: {logits.shape}")

        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # print(f"probs looks like: {probs.shape}")
        # sample from the distribution or take the most likely
        if sample:
            ix_mult = torch.multinomial(probs, num_samples=1)
            # print(f"action scaled by temperature: {ix_multinomial}")
            ix_argmax = torch.argmax(probs, dim=-1)  # simply return top argument
            # print(f"action picked by argmax: {ix}")
            ix = torch.topk(probs, k=num_recs, dim=-1)
            # print(ix_topk)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)
        x = ix
        # print(f"x.shape: {x.shape}")

    return x


def get_terminal_indices(arr):
    idxs = {}
    for i in reversed(range(len(arr))):
        idxs[arr[i]] = i
    done_idxs = list(idxs.values())
    done_idxs.reverse()
    done_idxs = done_idxs[1:]
    done_idxs.append(len(arr))
    # print(f"done_idxs were {done_idxs}")
    return done_idxs


def get_next_filename(figs_dir, base_filename):
    i = 1
    while True:
        new_filename = f"{base_filename}_{i}.svg"
        if not os.path.exists(os.path.join(figs_dir, new_filename)):
            return new_filename
        i += 1

def plot_reward_over_trajectory(rewards_over_trajectory, num_recs, user_id, epoch, max_epochs, num_users_to_eval, model_type, exp_number, figs_dir='figs/rewards_over_trajectory'):
    figs_dir = f'figs/rewards_over_trajectory/{exp_number}'

    plt.rcParams.update({'font.family': 'monospace'})
    plt.figure(figsize=(12, 8))

    # Plot rewards over trajectory
    plt.plot(range(1, num_recs + 1), rewards_over_trajectory, label=f'Rewards for User {user_id} (Epoch {epoch})',
             color='blue')

    # Centering x-axis at 0
    max_step = len(rewards_over_trajectory)
    plt.xlim(0, max_step * 1.1)

    # Centering y-axis at 0
    # max_abs_reward = np.max(np.abs(rewards_over_trajectory))
    max_abs_reward = 5
    plt.ylim(0, max_abs_reward * 1.1)

    # Calculate the line of best fit
    x = np.arange(1, len(rewards_over_trajectory) + 1)
    coeffs = np.polyfit(x, rewards_over_trajectory, 1)
    line_of_best_fit = np.poly1d(coeffs)
    plt.plot(x, line_of_best_fit(x), color='orange', linestyle='--')

    plt.xlabel('Recommendation Number', fontweight='bold', fontsize=14)
    plt.ylabel('Reward', fontweight='bold', fontsize=14)
    plt.title(f'Average Reward Over Trajectory for {num_users_to_eval} Randomly-Selected Users in Epoch {epoch} out of {max_epochs}', fontweight='bold', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    if figs_dir:
        os.makedirs(figs_dir, exist_ok=True)
        base_filename = f'rewards_per_trajectory_user_{user_id}_epoch_{epoch}_out_of_{max_epochs}'
        new_filename = get_next_filename(figs_dir, base_filename)
        plt.savefig(os.path.join(figs_dir, new_filename), format='svg')

    plt.close()


def plot_reward(rewards_per_epoch, context_length, batch_size, n_layer, n_head, n_embd,
              filename_train_dataset, len_train_dataset, learning_rate, lr_decay, num_users, num_recs, ratings_at_extreme, model_type, exp_number, figs_dir='figs'):
    plt.rcParams.update({'font.family': 'monospace'})
    plt.figure(figsize=(12, 8))

    # Plot average rewards per epoch
    plt.plot(range(1, len(rewards_per_epoch) + 1), rewards_per_epoch, label='Average Rewards per Epoch (Different User Every Time)', color='green')

    # Calculate the line of best fit
    x = np.arange(1, len(rewards_per_epoch) + 1)
    coeffs = np.polyfit(x, rewards_per_epoch, 1)
    line_of_best_fit = np.poly1d(coeffs)
    plt.plot(x, line_of_best_fit(x), color='orange', linestyle='--')

    if ratings_at_extreme:
        baseline = (num_recs * 0.65 * 5) + (num_recs * 0.35 * 1)  # 65% of ratings are 5 star
    else:
        baseline = (num_recs * 0.35 * 5) + (num_recs * 0.31 * 4) + (num_recs * 0.20 * 3) + (num_recs * 0.10 * 2) + (num_recs * 0.04 * 1)  # 65% of ratings are 5 star

    plt.axhline(y=baseline, color='blue', linestyle='--', label='Baseline (Random Choice)')

    plt.xlabel('Epoch', fontweight='bold', fontsize=14)
    plt.ylabel('Average Reward', fontweight='bold', fontsize=14)
    plt.title(f'Average Rewards Over {num_recs} New Recommendations Per Epoch', fontweight='bold', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Centering x-axis at 0
    max_epoch = len(rewards_per_epoch)
    plt.xlim(0, max_epoch * 1.1)

    # Centering y-axis at 0
    max_abs_reward = np.max(np.abs(rewards_per_epoch))
    plt.ylim(0, max_abs_reward * 1.1)

    # Information box
    info_text = f"Context Length: {context_length}\nBatch Size: {batch_size}\nLayers: {n_layer}\nHeads: {n_head}\nEmbedding Size: {n_embd}\nTrain Dataset Name: {filename_train_dataset}\nTrain Dataset Size: {len_train_dataset}\nLearning Rate: {learning_rate}\nLearning Rate Decay: {lr_decay}\nNumber of Users in Dataset: {num_users}\nNumber of recommendations given in series during evaluation: {num_recs}\nRatings at extreme (either 1 or 5): {ratings_at_extreme}\nModel Type: {model_type}"
    plt.text(0.02, 0.25, info_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom',
             bbox=dict(facecolor='white', alpha=0.8))

    if figs_dir:
        os.makedirs(figs_dir, exist_ok=True)
        base_filename = f'rewards_plot_with_info_{exp_number}'
        new_filename = get_next_filename(figs_dir, base_filename)
        plt.savefig(os.path.join(figs_dir, new_filename), format='svg')

    plt.close()

def plot_loss(train_losses, test_losses, context_length, batch_size, n_layer, n_head, n_embd,
              filename_train_dataset, len_train_dataset, filename_test_dataset, len_test_dataset, learning_rate, lr_decay, num_users, ratings_at_extreme, model_type, exp_number, figs_dir='figs'):
    plt.rcParams.update({'font.family': 'monospace'})
    plt.figure(figsize=(12, 8))

    # Plot training loss
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')

    if test_losses is not None:
        plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss', color='orange')

    plt.xlabel('Epoch', fontweight='bold', fontsize=14)
    plt.ylabel('Loss', fontweight='bold', fontsize=14)
    if test_losses is not None:
        plt.title('Training and Test Loss Over Epochs', fontweight='bold', fontsize=16)
    else:
        plt.title('Training Loss Over Epochs', fontweight='bold', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Centering x-axis at 0
    max_epoch = len(train_losses)
    plt.xlim(0, max_epoch * 1.1)

    # Centering y-axis at 0
    max_abs_reward = np.max(np.abs(train_losses))
    plt.ylim(0, max_abs_reward * 1.1)

    # Information box
    info_text = f"Context Length: {context_length}\nBatch Size: {batch_size}\nLayers: {n_layer}\nHeads: {n_head}\nEmbedding Size: {n_embd}\nTrain Dataset Name: {filename_train_dataset}\nTrain Dataset Size: {len_train_dataset}\nTest Dataset Name: {filename_test_dataset}\nTest Dataset Size: {len_test_dataset}\nLearning Rate: {learning_rate}\nLearning Rate Decay: {lr_decay}\nNo. Users in Dataset: {num_users}\nRatings at extreme (either 1 or 5): {ratings_at_extreme}\nModel type: {model_type}"
    plt.text(0.02, 0.85, info_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    if figs_dir:
        os.makedirs(figs_dir, exist_ok=True)
        base_filename = f'loss_plot_with_info_{exp_number}'
        new_filename = get_next_filename(figs_dir, base_filename)
        plt.savefig(os.path.join(figs_dir, new_filename), format='svg')

    plt.close()


def read_data(file_path):
    data = pd.read_csv(file_path, delimiter="\t")
    states = data['user_id'].tolist()
    actions = data['item_id'].tolist()
    actions = [a - 1 for a in actions]
    rewards = data['rating'].tolist()
    returns = []
    returns_to_go = np.zeros_like(rewards)
    timesteps = data['timestep'].tolist()
    terminal_indices = get_terminal_indices(states)  # Assuming get_terminal_indices is defined elsewhere
    start_index = 0

    if len(terminal_indices) > 1:
        # Generate returns-to-go
        for i in terminal_indices:
            rewards_by_episode = rewards[start_index:i]
            # print(f"rewards_by_episode: {rewards_by_episode} for data up to done_idx: {i}\n")
            returns = np.append(returns, sum(rewards_by_episode))
            for j in range(i - 1, start_index - 1, -1):
                rewards_by_reverse_growing_episode = rewards_by_episode[j - start_index:i - start_index]
                # print(f"rewards_by_reverse_growing_episode: {rewards_by_reverse_growing_episode}\n")
                returns_to_go[j] = sum(rewards_by_reverse_growing_episode)
                # print(f"returns_to_go: {returns_to_go}")
            start_index = i

        # print(f"states: {states}")
        # print(f"actions: {actions}")
        # print(f"returns_to_go: {returns_to_go}")

    return states, actions, rewards, returns, returns_to_go, timesteps, terminal_indices