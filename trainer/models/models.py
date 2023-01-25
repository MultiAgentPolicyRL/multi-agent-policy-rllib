import sys
import torch
import torch.nn as nn
from gym.spaces import Box, Dict
import numpy as np
from tensordict import TensorDict
from trainer.utils import exec_time

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

        self.MASK_NAME = "action_mask"
        # FIXME: this doesn't work with [22,22,22,22,22,22] of the planner
        self.num_outputs = action_space[0]
        self.logit_mask = torch.ones(self.num_outputs).to(self.device) * -10000000
        self.one_mask = torch.ones(self.num_outputs).to(self.device)
        ## TMP: parameters
        lr_actor = 0.001  # learning rate for actor network 0003
        lr_critic = 0.001  # learning rate for critic network 001

        # print(type(obs_space.spaces[self.MASK_NAME].shape))
        # mask = obs_space[self.MASK_NAME]
        self.mask_input = obs_space.spaces[self.MASK_NAME].shape

        # Fully connected values:
        self.fc_dim = 136
        self.num_fc = 2

        args = {"use_multiprocessing": False}

        # print(type(obs_space.spaces["flat"]))
        # sys.exit()
        # # flat = obs_space.spaces["flat"].shape
        self.actor = nn.Sequential(
            nn.Linear(
                get_flat_obs_size(obs_space.spaces["flat"]),
                self.num_outputs,  # , dtype=torch.float32
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
            obs2[key] = torch.from_numpy(obs[key]).to(self.device).detach()

        # obs1 =
        # obs['action_mask'] = torch.from_numpy(obs['action_mask']).to(self.device)#.detach()

        # obs1 = obs['flat'].squeeze()
        # obs['action_mask'] = obs['action_mask']
        action_probs = self.actor(obs2["flat"])

        # Apply logits mask
        logit_mask = self.logit_mask * (self.one_mask - obs["action_mask"])
        # logit_mask = torch.matmul(
        #     self.logit_mask, torch.sub(self.one_mask, obs2["action_mask"])
        # )

        action_probs = action_probs + logit_mask
        # action_probs = torch.add(logit_mask, action_probs)

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
        # TODO optimize memory so data doesn't need to be squeezed
        action_probs = self.actor(obs["flat"].squeeze())
        dist = torch.distributions.Categorical(action_probs)

        action_logprobs = dist.log_prob(act)
        dist_entropy = dist.entropy()
        state_values = self.critic(obs["flat"].squeeze())

        return action_logprobs, state_values, dist_entropy

    def forward(
        self,
    ):
        """
        Just don't.
        """
        NotImplementedError("Don't use this method.")

    def get_weights(self) -> dict:
        """
        Get policy weights.

        Return:
            actor_weights, critic_weights
        """
        actor_weights = self.actor.state_dict(keep_vars=False)
        # print(actor_weights)
        # print(type(actor_weights))
        # sys.exit()
        critic_weights = self.critic.state_dict(keep_vars=False)
        # optimizer_weights = self.optimizer.state_dict()
        optimizer_weights = 0
        return actor_weights, critic_weights, optimizer_weights

    def set_weights(self, actor_weights: dict, critic_weights: dict, optimizer_weights):
        """
        Set policy weights.

        Args:
            actor_weights: actor weights dictionary - from numpy
            critic_weights: critic weights dictionary - from numpy
        """
        # FIXME: optimizer weights are an issue!
        self.actor.load_state_dict(actor_weights)
        self.critic.load_state_dict(critic_weights)
        # self.optimizer.load_state_dict(optimizer_weights)
