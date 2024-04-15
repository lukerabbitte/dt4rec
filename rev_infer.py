"""
A proof-on-concept inference program, able to load a saved model checkpoint and recommend new items using the get_returns function from the trainer file

Also plots NDCG for each slate of recommendation at each timestep
"""


import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np
import random
from mingpt.rev_utils import sample
from mingpt.rev_model import GPT, GPTConfig
from mingpt.rev_utils import read_data
from torch.utils.data import Dataset
from mingpt.rev_utils import plot_reward_over_trajectory
import matplotlib.pyplot as plt
import os


class EvalDataset(Dataset):

    def __init__(self, states, actions, rewards, timesteps, terminal_indices, block_size):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.timesteps = timesteps
        self.terminal_indices = terminal_indices
        self.block_size = block_size
        self.vocab_size = max(actions) + 1

        self.start_indices = self.terminal_indices
        self.start_indices = np.insert(self.start_indices, 0, 0)

    def __len__(self):
        return len(self.states)

    # Returns user data from a complete matrix of user interactions where they have rated every item, for eval purposes
    def __getitem__(self, user_id):
        idx = self.start_indices[user_id - 1]
        done_idx = None if user_id == self.start_indices.size else self.terminal_indices[
            user_id - 1]  # avoid array out of limit bug for last user

        # Return tensors of (episode_length, 1)
        states = torch.tensor(np.array(self.states[idx:done_idx]), dtype=torch.float32).unsqueeze(1)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(self.rewards[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rewards, timesteps

model_type = 'return_to_go_conditioned'
num_recs = 30
num_users_to_eval = 1
context_length = 30
max_timestep = 141
eval_dataset_filename = 'data/goodreads_eval.tsv'
checkpoint_name = 'new_checkpoints/131/epoch_150_ckpt.pth'
infer_name = 'infer_20'

# Load eval dataset
eval_states, eval_actions, eval_rewards, _, _, eval_timesteps, eval_terminal_indices = read_data(eval_dataset_filename)
eval_dataset = EvalDataset(eval_states, eval_actions, eval_rewards, eval_timesteps, eval_terminal_indices, context_length * 3)
len_eval_dataset = len(eval_states)

if model_type == 'reward_only':
    ideal_return = 5
elif model_type == 'return_to_go_conditioned':
    ideal_return = num_recs * 5

# Load model
mconf = GPTConfig(273, 90, n_layer=6, n_head=8,
                  n_embd=128, model_type='return_to_go_conditioned', max_timestep=max_timestep)
model = GPT(mconf)
checkpoint = torch.load(checkpoint_name)
model.load_state_dict(checkpoint)
device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    model = torch.nn.DataParallel(model).to(device)
model.eval()

rewards_over_all_trajectories = []  # will be list of lists. For each inner list, find average and append this to average_rewards_over_all_trajectories
average_reward_over_all_trajectories_sum = 0

user_ids = random.sample(range(1, 256 + 1), num_users_to_eval)  # sample over num_users_to_eval users

for user_id in user_ids:

    all_ndcgs = []

    if model_type == 'reward_only':
        rewards = [ideal_return]
    elif model_type == 'return_to_go_conditioned':
        rtgs = [ideal_return]

    reward_sum = 0
    rewards_over_trajectory = []
    actions = []
    eval_states, eval_actions, eval_rewards, eval_timesteps = eval_dataset[user_id]

    for i in range(num_recs):

        state = torch.tensor([1., 0., 0., 0.])  # constant state
        state = state.unsqueeze(0).unsqueeze(0).to(device)

        # First sample with only state to begin
        if i == 0:
            first_actions = None
            if model_type == 'reward_only':
                first_rewards = torch.tensor(rewards, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1)
            elif model_type == 'return_to_go_conditioned':
                first_rtgs = torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1)
            first_timesteps = torch.ones((1, 1, 1), dtype=torch.int64).to(device)
            if model_type == 'reward_only':
                sampled_action = sample(model, state, first_actions, first_rewards, first_timesteps, 1,
                                        temperature=1.0, sample=True, num_recs=num_recs)
            elif model_type == 'return_to_go_conditioned':
                sampled_action = sample(model, state, first_actions, first_rtgs, first_timesteps, 1,
                                        temperature=1.0, sample=True, num_recs=num_recs)
            all_states = state

        actions_topk = sampled_action.indices.squeeze().tolist()
        print(f"Top k options for this rec were {actions_topk}")

        ### MEASURE A LIST OF THE TOP K OPTIONS AT EACH TIMESTEP, RATHER THAN JUST TAKING THE TOP ACTION NOT YET SEEN

        top_actions_not_yet_taken = [possible_action for possible_action in actions_topk if possible_action not in actions][:10]
        # print(f"10 best unseen actions for this rec were {top_actions_not_yet_taken}")

        action_indices = [np.where(eval_actions == action)[0][0] for action in top_actions_not_yet_taken]
        # print(f"Indices of 10 best unseen actions in eval_actions were {action_indices}")

        rewards_per_timestep = []
        for action_index in action_indices:
            reward_tensor = eval_rewards[action_index]
            reward = reward_tensor.item()
            rewards_per_timestep.append(reward)

        print(f"Rewards for the 10 best unseen actions were {rewards_per_timestep}")

        # Calculate DCG
        dcg = np.sum([reward / np.log2(i + 2) for i, reward in enumerate(rewards_per_timestep)])

        # Calculate IDCG
        ideal_rewards = sorted(rewards_per_timestep, reverse=True)
        idcg = np.sum([reward / np.log2(i + 2) for i, reward in enumerate(ideal_rewards)])

        # Calculate NDCG
        ndcg = dcg / idcg

        print(f"NDCG for the rewards is {ndcg}")

        all_ndcgs.append(ndcg)
        os.makedirs('../figs/ndcg', exist_ok=True)

        counter = 0


        # Define the parameters
        context_length = 30
        batch_size = 256
        n_layer = 6
        n_head = 8
        n_embd = 128
        filename_train_dataset = 'goodreads_data_first_2048_users_4_groups_beta_distribution.tsv'
        len_train_dataset = 99371
        learning_rate = 0.0006
        lr_decay = False
        num_users = 2048
        num_recs = 10
        model_type = 'return_to_go_conditioned'
        exp_number = 131

        # Define the info text
        info_text = f"Context Length: {context_length}\nBatch Size: {batch_size}\nLayers: {n_layer}\nHeads: {n_head}\nEmbedding Size: {n_embd}\nTrain Dataset Size: {len_train_dataset}\nLearning Rate: {learning_rate}\nLearning Rate Decay: {str(lr_decay)}\nNo. Users in Dataset: {num_users}\nModel type: {model_type}\nExperiment Number: {exp_number}"

        plt.figure(figsize=(10, 6))

        # Add the info text box
        plt.text(0.1, 0.5, info_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Set monospace font
        plt.rcParams['font.family'] = 'monospace'

        # Fit a line to the data
        if counter != 0:
            x = np.arange(len(all_ndcgs))
            coefficients = np.polyfit(x, all_ndcgs, 1)
            polynomial = np.poly1d(coefficients)

        # Plot the data
        plt.plot(all_ndcgs)

        # Plot the line of best fit
        if counter != 0:
            plt.plot(x, polynomial(x), label='Line of best fit', linestyle='--')

        # Set title with larger fontsize and bold font
        plt.title('NDCG Across Top Recommendations at Each Timestep (Trained Goodreads 100k Model)', fontsize=14, fontweight='bold')

        # Set x and y labels
        plt.xlabel('Timesteps')
        plt.ylabel('NDCG')

        plt.grid(True)

        # Set y-axis limits to center at 0
        plt.ylim(0, 1.1)

        plt.legend()

        while os.path.exists(f"figs/ndcg/ndcg_over_time_{counter}.svg"):
            counter += 1
        plt.savefig(f"figs/ndcg/ndcg_over_time_{counter}.svg")
        plt.close()

        ### TAKING JUST THE TOP ACTION NOT YET SEEN, RATHER THAN LOOKING AT THE TOP K NOT SEEN

        # Take most probable action from topk that isn't already in actions
        top_action_not_yet_taken = None
        for possible_action in actions_topk:
            if possible_action not in actions:
                top_action_not_yet_taken = possible_action
                break

        action = top_action_not_yet_taken
        print(f"The best action not already taken was {action + 1}")

        actions.append(action)
        action_indices = np.where(eval_actions == action)[0]

        if len(action_indices) > 0:
            action_index = action_indices[0]
        else:
            print(f"Action {action} was not found for user {eval_states[0]}")

        reward_tensor = eval_rewards[action_index]
        reward = reward_tensor.item()

        print(f"Action suggested for user {user_id} was item id {action + 1}, rated {reward} at index {action_index}")

        reward_sum += reward
        rewards_over_trajectory.append(reward)
        if model_type == 'reward_only':
            rewards.append(reward)
        elif model_type == 'return_to_go_conditioned':
            rtgs += [rtgs[-1] - reward]
        # print(f"rtgs is like {rtgs}")
        all_states = torch.cat([all_states, state], dim=1)
        all_actions = torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0)
        if model_type == 'reward_only':
            all_rewards = torch.tensor(rewards, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1)
        elif model_type == 'return_to_go_conditioned':
            all_rtgs = torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1)
        all_timesteps = (min(i + 1, max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device))

        if model_type == 'reward_only':
            sampled_action = sample(model, all_states, all_actions, all_rewards, all_timesteps, 1,
                                    temperature=1.0, sample=True, num_recs=num_recs)
        elif model_type == 'return_to_go_conditioned':
            sampled_action = sample(model, all_states, all_actions, all_rtgs, all_timesteps, 1,
                                    temperature=1.0, sample=True, num_recs=num_recs)
        rewards_over_all_trajectories.append(rewards_over_trajectory)  # list of lists
        # print(f"Just recommended {num_recs} new items to user {user_id} and the total rating was {reward_sum}")

rewards_over_all_trajectories = np.array(rewards_over_all_trajectories)
print(f"rewards_over_all_trajectories[0]: {rewards_over_all_trajectories[0]}")
print(f"rewards_over_all_trajectories[7]: {rewards_over_all_trajectories[7]}")
print(f"rewards_over_all_trajectories[9]: {rewards_over_all_trajectories[9]}")
average_rewards_over_all_trajectories = np.mean(rewards_over_all_trajectories, axis=0)
print(f"average_rewards_over_all_trajectories: {average_rewards_over_all_trajectories}")

print("\n")
print(all_ndcgs)
print("\n")

# Plot evolution of user ratings
plot_reward_over_trajectory(average_rewards_over_all_trajectories, num_recs, user_id, None, None, num_users_to_eval, model_type, 'infer_12')
average_reward_over_all_trajectories_sum = np.sum(average_rewards_over_all_trajectories)
print(f"average_reward_over_all_trajectories_sum: {average_reward_over_all_trajectories_sum}")