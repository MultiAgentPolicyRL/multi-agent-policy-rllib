import sys
import torch
import torch.nn as nn
from gym.spaces import Box, Dict
import numpy as np
from tensordict import TensorDict
from utils import exec_time

# pylint: disable=no-member


def get_flat_obs_size(obs_space):
    """
    Get flat observation size
    """
    if isinstance(obs_space, Box):
        return np.prod(obs_space.shape)
    elif not isinstance(obs_space, Dict):
        raise TypeError

    def rec_size(obs_dict_space, n=0):
        for subspace in obs_dict_space.spaces.values():
            if isinstance(subspace, Box):
                n = n + np.prod(subspace.shape)
            elif isinstance(subspace, Dict):
                n = rec_size(subspace, n=n)
            else:
                raise TypeError
        return n

    return rec_size(obs_space)


def apply_logit_mask1(logits, mask):
    """Mask values of 1 are valid actions."
    " Add huge negative values to logits with 0 mask values."""
    logit_mask = torch.ones(logits.shape) * -10000000
    logit_mask = logit_mask * (1 - mask)

    return logits + logit_mask


class PytorchLinear(nn.Module):
    """A linear (feed-forward) model."""

    def __init__(self, obs_space, action_space, device):
        super().__init__()
        self.device = device
        self.logit_mask = torch.ones(50).to(self.device) * -10000000

        ## TMP: parameters
        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.001  # learning rate for critic network

        self.MASK_NAME = "action_mask"
        self.num_outputs = action_space

        mask = obs_space[self.MASK_NAME]
        self.mask_input = mask.shape

        # Fully connected values:
        self.fc_dim = 136
        self.num_fc = 2

        args = {"use_multiprocessing": False}

        self.actor = nn.Sequential(
            nn.Linear(
                get_flat_obs_size(obs_space["flat"]), 50  # , dtype=torch.float32
            ),
            nn.ReLU(),
            # nn.Linear(32, self.num_outputs),
        )

        self.fc_layers_val_layers = []  # nn.Sequential()

        for _ in range(self.num_fc):
            self.fc_layers_val_layers.append(nn.Linear(self.fc_dim, self.fc_dim))
            self.fc_layers_val_layers.append(nn.ReLU())

        self.fc_layers_val_layers.append(nn.Linear(self.fc_dim, 1))

        self.critic = nn.Sequential(*self.fc_layers_val_layers)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": lr_actor},
                {"params": self.critic.parameters(), "lr": lr_critic},
            ]
        )

    # @exec_time
    def act(self, obs):
        """
        Args:
            obs: agent environment observation

        Returns:
            action: taken action
            action_logprob: log probability of that action
        """
        obs2 = {}
        for key in obs.keys():
            obs2[key] = torch.from_numpy(obs[key]).to(self.device)

        obs1 = obs2['flat'].squeeze()
        obs['action_mask'] = torch.from_numpy(obs['action_mask']).to(self.device).detach()

        action_probs = self.actor(obs1)

        # Apply logits mask
        logit_mask = self.logit_mask
        logit_mask = logit_mask * (1 - obs["action_mask"].squeeze(0))
        action_probs = action_probs + logit_mask

        dist = torch.distributions.Categorical(logits=action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, obs, act):
        """
        Args:
            obs: agent environment observation
            act: action that is mapped with

        Returns:
            action_logprobs: log probability that `act` is taken with this model
            state_values: value function reward prediction
            dist_entropy: entropy of actions distribution
        """
        action_probs = self.actor(obs["flat"].squeeze().float())
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(act)
        dist_entropy = dist.entropy()
        state_values = self.critic(obs["flat"].squeeze().float())

        return action_logprobs, state_values, dist_entropy

    def forward(
        self,
    ):
        """
        Just don't.
        """
        NotImplementedError("Don't use this method.")

    def get_weights(self):
        """
        Get policy weights.

        Return:
            actor_weights, critic_weights
        """
        # FIXME: add return type
        actor_weights = self.actor.state_dict()
        critic_weights = self.critic.state_dict()
        optimizer_weights = self.optimizer.state_dict()

        return actor_weights, critic_weights, optimizer_weights

    def set_weights(self, actor_weights, critic_weights, optimizer_weights):
        """
        Set policy weights.
        """
        # FIXME: docs
        # FIXME: add args type
        self.actor.load_state_dict(actor_weights)
        self.critic.load_state_dict(critic_weights)
        self.optimizer.load_state_dict(optimizer_weights)
