"""
Main training and evaluation loop
"""

import math
import logging

import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from mingpt.rev_utils import sample
from mingpt.rev_utils import plot_reward_over_trajectory
import random
import os

logger = logging.getLogger(__name__)

class MatchingActionNotFoundError(Exception):
    def __init__(self, message="Matching action not found in eval_actions."):
        self.message = message
        super().__init__(self.message)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, eval_dataset, eval_data):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.eval_dataset = eval_dataset
        self.eval_data = eval_data
        self.train_losses = []
        self.test_losses = []
        self.action_losses = []
        self.rewards_per_epoch = []

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self, epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_name = f"epoch_{epoch}_ckpt.pth"
        ckpt_path = os.path.join(self.config.ckpt_dir, ckpt_name)
        os.makedirs(self.config.ckpt_dir, exist_ok=True)
        logger.info("saving %s", ckpt_path)
        torch.save(raw_model.state_dict(), ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        test_actions = []

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            # print(f"is_train: {is_train}")
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            action_losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            # Note that x,y,r are of size torch.Size([30]), or context length, and t of size ([1])
            for it, (x, y, r, t) in pbar:

                rewards_per_iteration = []

                # print(f"about to print t of shape: {t.shape}")
                # print(t)

                # print(f"about to print y of shape: {y.shape}")
                # print(y)

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # print(f"Train?: {is_train} - what does x of size: {x.size()} look like?: x[1]")  # ([128, 30, 1])
                # print(f"Train?: {is_train} - what does y of size: {y.size()} look like?: y[1]")  # ([128, 30, 1])
                # print(f"Train?: {is_train} - what does r of size: {r.size()} for user: {x[1]} and actions: {y[1]} look like?: {r[1]}")  # ([128, 30, 1])
                # print(f"Train?: {is_train} - what does t of size: {t.size()} look like?: t[1]")  # ([128, 1, 1])

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss, action_loss = model(x, y, y, r, t)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    action_loss = action_loss.mean()
                    action_losses.append(action_loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

                    # print(f"x[-1].shape is: {x[-1].shape}")  # ([30, 1])

                    # lets sample
                    # sampled_action = sample(model, x[-1].unsqueeze(0), 1, temperature=1.0, sample=True, top_k=None, actions=y[-1].unsqueeze(0), rtgs=r[-1].unsqueeze(0), timesteps=t[-1].unsqueeze(0))
                    # print(f"sampled_action was: {sampled_action.squeeze(0).squeeze(0) + 1} for user: {x[-1][1].item()}")
                    # print(sampled_action.shape)

            print("printed entire dataset")

            if is_train:
                self.train_losses.append(float(np.mean(losses)))
                print(f"train_loss is: {float(np.mean(losses))}")
                self.action_losses.append(float(np.mean(action_losses)))
                # print(f"action_loss is: {float(np.mean(action_losses))}")

            if not is_train:
                test_loss = float(np.mean(losses))
                self.test_losses.append(test_loss)
                print(f"test_loss is: {float(np.mean(losses))}")
                logger.info("test loss: %f", test_loss)

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay

        self.save_checkpoint(-1)

        for epoch in range(config.max_epochs):
            print(f"entering epoch loop")
            run_epoch('train')

            if self.test_dataset is not None:
                run_epoch('test')

            """
            # Do basic evaluation
            states = [1]
            state = torch.tensor(states, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1)
            print(f"state shape is {state.shape}")
            rewards = [5]
            first_rewards = torch.tensor(rewards, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1)
            first_actions = None
            first_timesteps = torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)
            num_recs = 20

            sampled_action = sample(self.model, state, first_actions, first_rewards, first_timesteps, 1,
                                                temperature=1.0, sample=True, num_recs=num_recs)

            actions_topk = sampled_action.indices.squeeze().tolist()

            top_action_not_yet_taken = None
            for possible_action in actions_topk:
                if possible_action not in test_actions:
                    top_action_not_yet_taken = possible_action
                    break


            action = top_action_not_yet_taken
            test_actions.append(action)
            actions_topk = [a+1 for a in actions_topk]

            print(actions_topk)
            print(f"best action not yet taken was {action + 1}")
            """


            # Evaluate by passing in the number of new recs we want to generate.
            reward_per_epoch = self.get_returns(self.config.num_recs, self.config.num_users, self.config.ratings_per_user, epoch, config.max_epochs, self.config.model_type, self.config.exp_number, self.config.max_return)
            self.rewards_per_epoch.append(reward_per_epoch)


        self.save_checkpoint(config.max_epochs)
        return self.train_losses, self.action_losses, self.test_losses, self.rewards_per_epoch


    def get_returns(self, num_recs, num_users, ratings_per_user, epoch, max_epochs, model_type, exp_number, max_return):

        # ideal_return = num_recs * 5  # condition sequence on 'command' or desired return + time horizon
        print(f"model type received in get_returns: {model_type}")
        if model_type == 'reward_only':
            ideal_return = 5  # TODO rtg/reward
        elif model_type == 'return_to_go_conditioned':
            ideal_return = num_recs * 5  # TODO rtg/reward
            # ideal_return = max_return
        self.model.train(False)
        all_rewards_over_trajectory = []  # will be list of lists. For each inner list, find average and append this to average_rewards_over_trajectory
        sum_average_rewards_over_trajectory = 0

        # num_users_to_eval = 50
        # user_ids = random.sample(range(1, 256 + 1), num_users_to_eval)  # sample over num_users_to_eval users

        user_ids = [158] # can change to reflect average users
        num_users_to_eval = 1

        for user_id in user_ids:

            if model_type == 'reward_only':
                rewards = [ideal_return]  # TODO rtgs/rewards
            elif model_type == 'return_to_go_conditioned':
                rtgs = [ideal_return] # TODO rtgs/rewards
            reward_sum = 0
            rewards_over_trajectory = []
            actions = []
            eval_states, eval_actions, eval_rewards, eval_timesteps = self.eval_dataset[user_id]
            # print(f"rtgs is like {rtgs}")

            for i in range(num_recs):

                # state = torch.tensor([1.])  # constant state
                state = torch.tensor([1., 0., 0., 0.])
                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                # print(f"state unsqueezed was: {state.shape}")

                # def sample(model, x, y, r, t, steps, temperature=1.0, sample=False, top_k=None):
                # First sample with only state to kick us off
                if i == 0:
                    first_actions = None
                    if model_type == 'reward_only':
                        first_rewards = torch.tensor(rewards, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(
                            -1) # TODO rtg/reward
                    elif model_type == 'return_to_go_conditioned':
                        first_rtgs = torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1) # TODO rtg/reward
                    first_timesteps = torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)
                    if model_type == 'reward_only':
                        sampled_action = sample(self.model, state, first_actions, first_rewards, first_timesteps, 1, temperature=1.0, sample=True, num_recs=num_recs) # TODO rtg/reward
                    elif model_type == 'return_to_go_conditioned':
                        sampled_action = sample(self.model, state, first_actions, first_rtgs, first_timesteps, 1,
                                                temperature=1.0, sample=True, num_recs=num_recs)  # TODO rtg/reward
                    all_states = state

                
                # action = sampled_action.cpu().numpy()[-1][0]
                actions_topk = sampled_action.indices.squeeze().tolist()
                # print(f"top k options for this rec were {actions_topk}")


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



                ### TAKING JUST THE TOP ACTION NOT YET SEEN, RATHER THAN LOOKING AT THE TOP K NOT SEEN

                # Take leftmost action from topk that isn't already in actions
                top_action_not_yet_taken = None
                for possible_action in actions_topk:
                    if possible_action not in actions:
                        top_action_not_yet_taken = possible_action
                        break

                action = top_action_not_yet_taken

                # print(f"The best action not already taken was {action}")
                # if action in actions:
                #     print(f"Major alert, action {action} was already in actions!")

                actions.append(action)
                action_indices = np.where(eval_actions == action)[0]

                if len(action_indices) > 0:
                    action_index = action_indices[0]
                else:
                    print(f"Action {action} was not found for user {eval_states[0]}")
                    raise MatchingActionNotFoundError()

                reward_tensor = eval_rewards[action_index]
                reward = reward_tensor.item()

                # print(f"Action suggested for user {user_id} was item id {action}, rated {reward} at index {action_index}")

                reward_sum += reward
                rewards_over_trajectory.append(reward)
                if model_type == 'reward_only':
                    rewards.append(reward) # TODO rtg/reward
                    # rewards.append(5)
                elif model_type == 'return_to_go_conditioned':
                    rtgs += [rtgs[-1] - reward] # TODO rtg/reward
                # print(f"rtgs is like {rtgs}")
                all_states = torch.cat([all_states, state], dim=1) # TODO try without cat
                all_actions = torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0)
                if model_type == 'reward_only':
                    all_rewards = torch.tensor(rewards, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1) # TODO rtg/reward
                elif model_type == 'return_to_go_conditioned':
                    all_rtgs = torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(
                        -1)  # TODO rtg/reward
                all_timesteps = (min(i+1, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device))
                # print(f"all_actions shape just before sample {all_actions.shape}")
                
                # print(all_states)
                # print(all_actions)
                # print(all_rtgs)
                # print(all_timesteps)
                
                if model_type == 'reward_only':
                    sampled_action = sample(self.model, all_states, all_actions, all_rewards, all_timesteps, 1, temperature=1.0, sample=True, num_recs=num_recs) # TODO rtg/reward
                elif model_type == 'return_to_go_conditioned':
                    sampled_action = sample(self.model, all_states, all_actions, all_rtgs, all_timesteps, 1,
                                            temperature=1.0, sample=True, num_recs=num_recs) # TODO rtg/reward
                all_rewards_over_trajectory.append(rewards_over_trajectory) # list of lists
                # print(f"Just recommended {num_recs} new items to user {user_id} and the total rating was {reward_sum}")

        all_rewards_over_trajectory = np.array(all_rewards_over_trajectory)
        average_rewards_over_trajectory = np.mean(all_rewards_over_trajectory, axis=0)
        print(f"average_rewards_over_trajectory: {average_rewards_over_trajectory}")


        self.model.train(True)
        # Plot evolution of user ratings
        plot_reward_over_trajectory(average_rewards_over_trajectory, num_recs, None, epoch, max_epochs, num_users_to_eval, model_type, exp_number)
        sum_average_rewards_over_trajectory = np.sum(average_rewards_over_trajectory)
        print(f"sum_average_rewards_over_trajectory: {sum_average_rewards_over_trajectory}")

        return sum_average_rewards_over_trajectory
