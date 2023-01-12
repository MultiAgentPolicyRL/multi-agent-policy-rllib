import os
import torch
import logging
import numpy as np
import torch.nn as nn
from typing import Union
from torch.optim import Adam

from ai_economist_ppo_dt.utils import get_basic_logger, time_it


class Actor(nn.Module):
    """
    Actor (Policy) Model.
    =====


    """

    def __init__(
        self,
        output_size: int = 50,
        conv_first_dim: tuple = (7, 2),
        conv_filters: tuple = (16, 32),
        filter_size: int = 3,
        log_level: int = logging.INFO,
        log_path: str = None,
        device: str = "cpu",
    ) -> None:
        """
        Initialize parameters and build model.

        """

        self.output_size = output_size
        self.device = device
        self.logger: logging = get_basic_logger(
            "Actor", level=log_level, log_path=log_path
        )

        super(Actor, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(conv_first_dim[0], conv_filters[0], filter_size).to(
            device
        )
        self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], filter_size).to(device)
        self.flatten = nn.Flatten().to(device)

        # self.concat = torch.cat([self.conv1, self.conv2, self.flatten])
        self.dense1 = nn.Linear(1704, 128).to(
            device
        )  # Before was 32*32*32 but now hard-coded to 1704, to understand why
        self.dense2 = nn.Linear(128, 128).to(device)

        self.relu = nn.ReLU(inplace=False).to(device)
        self.sigmoid = nn.Sigmoid().to(device)

        self.LSTM = nn.LSTM(128, 128, 1, batch_first=True).to(device)
        self.reshape = nn.Linear(128, output_size).to(device)

        self.optimizer = Adam(self.parameters(), lr=0.0003)

    # @time_it
    def forward(self, state: Union[np.ndarray, list]) -> torch.FloatTensor:
        """
        Build a network that maps state -> action probabilities.
        """
        self.logger.debug(f"state: ({len(state)},)")
        world_map = state[0]
        self.logger.debug(f"world_map: {world_map.shape}")

        flat = state[1]
        self.logger.debug(f"flat: {flat.shape}")

        # Convolutional layers
        x = self.relu(self.conv1(world_map))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)

        # Concatenate
        x = torch.cat([x, flat], dim=1)

        # Dense layers
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))

        # LSTM
        x, _ = self.LSTM(x)

        # Sigmoid
        x = self.sigmoid(x)

        # Reshape and return
        x = self.reshape(x)
        x.to(self.device)
        self.logger.debug(f"output: {x.shape}")

        return x

    # @time_it
    def my_loss(self, output, target) -> torch.FloatTensor:
        """
        Custom loss from [arxiv:1707.06347](https://arxiv.org/abs/1707.06347).
        """
        # Calculate the loss
        advantages = target[0]
        predictions = target[1]
        actions = target[2]

        # Define constants
        clipping = 0.2
        entropy_loss = 0.001

        prob = actions * output
        old_prob = actions * predictions

        prob = torch.clamp(prob, 1e-10, 1.0)
        old_prob = torch.clamp(old_prob, 1e-10, 1.0)

        ratio = torch.exp(torch.log(prob) - torch.log(old_prob))

        p1 = ratio * advantages
        p2 = torch.clamp(ratio, 1 - clipping, 1 + clipping) * advantages

        actor_loss = -torch.mean(torch.min(p1, p2))

        entropy = -(output * torch.log(output + 1e-10))
        entropy = entropy_loss * torch.mean(entropy)

        loss = actor_loss
        if not torch.isnan(entropy):
            loss = actor_loss + entropy
        loss.requires_grad_(True)

        return loss
