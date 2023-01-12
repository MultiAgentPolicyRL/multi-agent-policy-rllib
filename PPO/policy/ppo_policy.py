"""
docs about this file
"""
import copy
import logging
import random
import sys
import time

import numpy as np

# import tensorflow as tf
import torch

# from model.model import LSTMModel
from model.model import PytorchLinear as LSTMModel
from policy.ppo_policy_config import PpoPolicyConfig
from utils.timeait import timeit
from memory import RolloutBuffer


class PPOAgent:
    """
    PPO Main Optimization Algorithm
    """

    def __init__(self, policy_config: PpoPolicyConfig):
        # Initialization
        # Environment and PPO parameters
        self.policy_config = policy_config
        self.action_space = self.policy_config.action_space  # self.env.action_space.n
        self.batch_size = self.policy_config.batch_size  # training epochs

        # self.Model: LSTMModel = LSTMModel(policy_config.model_config)
        self.Model: LSTMModel = LSTMModel(
            obs_space=policy_config.model_config.observation_space,
            action_space=policy_config.model_config.action_space,
            num_outputs=policy_config.model_config.action_space,
            model_config=None,
            name=None,
        )

    # @timeit
    def act(self, observation: dict):
        """
        Given an observation, returns `policy_action`, `policy_probability` and `vf_action` from the model.
        In this case (PPO) it's just a reference to call the model's forward method ->
        it's an "exposed API": common named functions for each policy.

        Args:
            observation: single agent observation of the environment.

        Returns:
            policy_action: predicted action(s)
            policy_probability: action probabilities
            vf_action: value function action predicted
        """
        # torch.squeeze(logits)

        # Get the prediction from the Actor network
        with torch.no_grad():
            policy_action, policy_probability, vf_action = self.Model.act(observation)

        return policy_action.item(), policy_probability, vf_action

    def learn(
        self,
        rolloutBuffer: RolloutBuffer,
        epochs: int,
        steps_per_epoch: int,
    ):
        """
        Train Policy networks
        Takes as input the batch with N epochs of M steps_per_epoch. As we are using an LSTM
        model we are not shuffling all the data to create the minibatch, but only shuffling
        each epoch.

        Example:
            Input epochs: 0,1,2,3
            Shuffled epochs: 2,0,1,3

        It calls `self.Model.fit` passing the shuffled epoch.

        INFO: edit this function to modify how minibatch creation is managed.
        INFO: there's a simpler way to do the same, but memory need to be adapted for it to work.

            data = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
            epochs = 4
            steps_per_epoch = 5
            for i in range(epochs):
                print(dati[:i:steps_per_epoch])

        Args:
            observations: Agent ordered, time listed observations per agent
            policy_actions: Agent ordered, time listed policy_actions per agent
            policy_probabilitiess: Agent ordered, time listed policy_probabilitiess per agent
            value_functions: Agent ordered, time listed observalue_functionsvations per agent
            rewards: Agent ordered, time listed rewards per agent
            epochs: how many epochs in the given batch (it is equal to n_agents in the selected
            batch)
            steps_per_epoch: how long is the epoch (it's equal to algorithm_config.batch_size)

        Returns:
            nothing

        """

        """
        Logic simplified:
            epochs = 4
            batch_size = 5
            epochs_selected = list(range(epochs))
            random.shuffle(epochs_selected)
            data = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
            for i in epochs_selected:
                print(data[i*batch_size:batch_size+i*batch_size])
        """
        # Set epochs order
        # epochs_order = list(range(10))
        # steps_per_epoch = int(400)

        epochs_order = list(range(epochs))
        steps_per_epoch = int(steps_per_epoch)
        random.shuffle(epochs_order)

        temp = list(
            zip(
                observations,
                policy_actions,
                policy_probabilitiess,
                value_functions,
                rewards,
            )
        )
        random.shuffle(temp)
        (
            observations,
            policy_actions,
            policy_probabilitiess,
            value_functions,
            rewards,
        ) = zip(*temp)

        for i in epochs_order:
            # Get it's data
            selected_observations = observations[
                i * steps_per_epoch : steps_per_epoch + i * steps_per_epoch
            ]
            selected_policy_actions = policy_actions[
                i * steps_per_epoch : steps_per_epoch + i * steps_per_epoch
            ]
            selected_policy_probabilitiess = policy_probabilitiess[
                i * steps_per_epoch : steps_per_epoch + i * steps_per_epoch
            ]
            selected_value_functions = value_functions[
                i * steps_per_epoch : steps_per_epoch + i * steps_per_epoch
            ]
            selected_rewards = rewards[
                i * steps_per_epoch : steps_per_epoch + i * steps_per_epoch
            ]

            self.update(rolloutBuffer)
            # Fit the model
            # self.Model.fit(
            #     observations=selected_observations,
            #     policy_actions=selected_policy_actions,
            #     policy_probabilities=selected_policy_probabilitiess,
            #     vf_actions=selected_value_functions,
            #     rewards=selected_rewards,
            # )

    def update(self, buffer: RolloutBuffer):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(
            reversed(buffer.rewards), reversed(buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(buffer.logprobs, dim=0)).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.Model.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio pi_theta / pi_theta_old
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.Model.optimizer.zero_grad()
            loss.mean().backward()
            self.Model.optimizer.step()
