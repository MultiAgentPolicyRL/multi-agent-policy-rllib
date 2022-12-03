import os
import torch
import logging
import numpy as np
import torch.nn as nn
from typing import Union
from torch.optim import Adam

from ai_economist_ppo_dt.utils import get_basic_logger, time_it


class Critic(nn.Module):
    """
    Critic (Policy) Model.
    =====


    """
    def __init__(self, conv_first_dim: tuple = (7, 2), conv_filters: tuple = (16,32), filter_size: int = 3, log_level: int = logging.INFO, log_path: str = None,  device: str = "cpu"):
        """
        Initialize parameters and build model.

        """
        self.device = device
        self.logger:logging = get_basic_logger("Critic", level=log_level, log_path=log_path)

        super(Critic, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(conv_first_dim[0], conv_filters[0], filter_size).to(device)  
        self.conv2 = nn.Conv2d(conv_filters[0], conv_filters[1], filter_size).to(device)
        self.flatten = nn.Flatten().to(device)

        #self.concat = torch.cat([self.conv1, self.conv2, self.flatten])
        self.dense1 = nn.Linear(1704, 128).to(device) # Before was 32*32*32 but now hard-coded to 1704, to understand why
        self.dense2 = nn.Linear(128, 128).to(device)
        # self.reshape = nn.Linear(128, output_size)
        
        self.relu = nn.ReLU().to(device)
        self.sigmoid = nn.Sigmoid().to(device)

        self.LSTM = nn.LSTM(128, 128, 1, batch_first=True).to(device)
        self.dense3 = nn.Linear(128, 1).to(device)

        self.optimizer = Adam(self.parameters(), lr=0.0003)

    # @time_it
    def forward(self, state: Union[np.ndarray, list]) -> torch.Tensor:
        """
        Build a network that maps state -> action probabilities.
        """
        self.logger.debug(f"state: {len(state)}")
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
        # x = self.reshape(x)

        # LSTM
        x, _ = self.LSTM(x)

        # Output layer
        x = self.dense3(x)
        x.to(self.device)

        self.logger.debug(f"output: {x.shape}")

        return x

    def my_loss(self, output, target):
        values = target[1]
        target = target[0]
        
        #return torch.mean((target - output) ** 2)

        # L_CLIP
        LOSS_CLIPPING = 0.2
        # clipped_value_loss = values + K.clip(
        #     y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING
        # )
        clipped_value_loss = values + torch.clamp(
            output - values, -LOSS_CLIPPING, LOSS_CLIPPING
        )
        v_loss1 = (target - clipped_value_loss) ** 2
        v_loss2 = (target - output) ** 2
        value_loss = 0.5 * torch.mean(torch.maximum(v_loss1, v_loss2))

        value_loss.requires_grad_(True)

        return value_loss
