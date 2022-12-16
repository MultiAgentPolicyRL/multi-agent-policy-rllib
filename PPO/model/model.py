"""
LSTM NN Model for AI-Economist RL environment
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=unused-import

import sys

import numpy as np
import torch
import torch.nn as nn
from model.model_config import ModelConfig

WORLD_MAP = "world-map"
WORLD_IDX_MAP = "world-idx_map"
ACTION_MASK = "action_mask"


def apply_logit_mask(logits, mask):
    """
    Apply mask to logits, gets an action and calculates its log_probability.
    Mask values of 1 are valid actions.

    Args:
        logits: actions probability distribution
        mask: action_mask (consists of a tensor of N boolean [0,1] values)

    Returns:
        action: predicted action
        probs: this `action` log_probability
    """

    # Add huge negative values to logits with 0 mask values.
    logit_mask = torch.ones(logits.shape) * -10000000
    logit_mask = logit_mask * (1 - mask)
    logit_mask = logits + logit_mask

    # Softmax is used to have sum(logit_mask) == 1 -> so it's a probability distibution
    logit_mask = torch.softmax(logit_mask, dim=1)
    # Makes a Categorical distribution
    dist = torch.distributions.Categorical(logit_mask)
    # Gets the action
    action = dist.sample()
    # Gets action log_probability
    probs = torch.squeeze(dist.log_prob(action)).item()

    return action, probs


class LSTMModel(nn.Module):
    """
    policy&value_function (Actor-Critic) Model
    =====
    """

    def __init__(self, modelConfig: ModelConfig) -> None:
        """
        Initialize the policy&value_function Model.
        """
        super(LSTMModel, self).__init__()
        self.model_config = modelConfig
        # self.logger = get_basic_logger(name, level=log_level, log_path=log_path)
        self.shapes = dict()

        # This is for managing all the possible inputs without having more networks
        for key, value in self.model_config.observation_space.items():
            # Check if the input must go through a Convolutional Layer
            if key == ACTION_MASK:
                pass
            elif key == WORLD_MAP:
                self.conv_shape_r, self.conv_shape_c, self.conv_map_channels = (
                    value.shape[1],
                    value.shape[2],
                    value.shape[0],
                )
            elif key == WORLD_IDX_MAP:
                self.conv_idx_channels = value.shape[0] * self.model_config.emb_dim
        ###

        self.embed_map_idx = nn.Embedding(
            self.model_config.input_emb_vocab,
            self.model_config.emb_dim,
            device=self.model_config.device,
            dtype=torch.float32,
        )
        self.conv_layers = nn.ModuleList()
        self.conv_shape = (
            self.conv_shape_r,
            self.conv_shape_c,
            self.conv_map_channels + self.conv_idx_channels,
        )

        for i in range(1, self.model_config.num_conv):
            if i == 1:
                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=self.conv_shape[1],
                        out_channels=self.model_config.filter[0],
                        kernel_size=self.model_config.kernel_size,
                        stride=self.model_config.strides,
                        # padding_mode='same',
                    )
                )
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=self.model_config.filter[0],
                    out_channels=self.model_config.filter[1],
                    kernel_size=self.model_config.kernel_size,
                    stride=self.model_config.strides,
                    # padding_mode='same',
                )
            )

        self.conv_dims = (
            self.model_config.kernel_size[0]
            * self.model_config.strides
            * self.model_config.filter[1]
        )
        self.flatten_dims = (
            self.conv_dims
            + self.model_config.observation_space["flat"].shape[0]
            + len(self.model_config.observation_space["time"])
        )
        self.fc_layer_1 = nn.Linear(
            in_features=self.flatten_dims, out_features=self.model_config.fc_dim
        )
        self.fc_layer_2 = nn.Linear(
            in_features=self.model_config.fc_dim, out_features=self.model_config.fc_dim
        )
        self.lstm = nn.LSTM(
            input_size=self.model_config.fc_dim,
            hidden_size=self.model_config.cell_size,
            num_layers=1,
        )
        self.layer_norm = nn.LayerNorm(self.model_config.fc_dim)
        self.output_policy = nn.Linear(
            in_features=self.model_config.cell_size,
            out_features=self.model_config.output_size,
        )
        self.output_value = nn.Linear(
            in_features=self.model_config.cell_size, out_features=1
        )

        self.relu = nn.ReLU()
        self.hidden_state_h_p = torch.zeros(
            1, self.model_config.cell_size, device=self.model_config.device
        )
        self.hidden_state_c_p = torch.zeros(
            1, self.model_config.cell_size, device=self.model_config.device
        )
        self.hidden_state_h_v = torch.zeros(
            1, self.model_config.cell_size, device=self.model_config.device
        )
        self.hidden_state_c_v = torch.zeros(
            1, self.model_config.cell_size, device=self.model_config.device
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.model_config.lr)

        # self.logger.info("Model created successfully")

    # @time_it
    def forward(self, observation: dict):
        """
        Model's forward. Given an agent observation, action distribution and value function
        prediction are returned.

        Args:
            observation: agent observation

        Returns:
            policy_action: action taken by the actor, example: Tensor([2])
            policy_probabiliy: `policy_action` log_probability
            vf_prediction: value function action prediction
        """
        if isinstance(observation, dict):
            _world_map = observation[WORLD_MAP]
            _world_idx_map = observation[WORLD_IDX_MAP]
            _flat = observation["flat"]
            _time = observation["time"]
            _action_mask = observation[ACTION_MASK]
        else:
            _world_map = observation[0]
            _world_idx_map = observation[1].long()
            _flat = observation[2]
            _time = observation[3]
            _action_mask = observation[4]

        if self.model_config.name == "p":
            _p0 = observation["p0"]
            _p1 = observation["p1"]
            _p2 = observation["p2"]
            _p3 = observation["p3"]

        conv_input_map = torch.permute(_world_map, (0, 2, 3, 1))
        conv_input_idx = torch.permute(_world_idx_map, (0, 2, 3, 1))

        # Concatenate the remainings of the input
        if self.model_config.name == "p":
            non_convolutional_input = torch.cat(
                [
                    _flat,
                    _time,
                    _p0,
                    _p1,
                    _p2,
                    _p3,
                ],
                axis=1,
            )
        else:
            non_convolutional_input = torch.cat(
                [
                    _flat,
                    _time,
                ],
                axis=1,
            )

        for tag in ["_policy", "_value"]:
            # Embedd from 100 to 4
            map_embedd = self.embed_map_idx(
                conv_input_idx
            )  # TO CHECK WHICH IS THE INPUT -- DONE
            # Reshape the map
            map_embedd = torch.reshape(
                map_embedd,
                (-1, self.conv_shape_r, self.conv_shape_c, self.conv_idx_channels),
            )
            # Concatenate the map and the idx map
            conv_input = torch.cat([conv_input_map, map_embedd], axis=-1)
            # Convolutional Layers
            for conv_layer in self.conv_layers:
                conv_input = self.relu(conv_layer(conv_input))
            # Flatten the output of the convolutional layers
            flatten = torch.reshape(
                conv_input, (-1, self.conv_dims)
            )  # 192 is from 32 * 3 * 2
            # Concatenate the convolutional output with the non convolutional input
            fc_in = torch.cat([flatten, non_convolutional_input], axis=-1)
            # Fully Connected Layers
            for i in range(self.model_config.num_fc):
                if i == 0:
                    fc_in = self.relu(self.fc_layer_1(fc_in))
                else:
                    fc_in = self.relu(self.fc_layer_2(fc_in))
            # Normalize the output
            layer_norm_out = self.layer_norm(fc_in)
            # LSTM

            # Project LSTM output to logits or value
            if tag == "_policy":
                lstm_out, hidden = self.lstm(
                    layer_norm_out, (self.hidden_state_h_p, self.hidden_state_c_p)
                )
                self.hidden_state_h_p, self.hidden_state_c_p = hidden
                policy_action, policy_probability = apply_logit_mask(
                    self.output_policy(lstm_out), _action_mask
                )
            else:
                lstm_out, hidden = self.lstm(
                    layer_norm_out, (self.hidden_state_h_v, self.hidden_state_c_v)
                )
                self.hidden_state_h_v, self.hidden_state_c_v = hidden
                value = self.output_value(lstm_out)

        return policy_action, policy_probability, value

    def fit(
        self,
        observations,
        policy_actions,
        policy_probabilitiess,
        value_functions,
        rewards,
    ):
        """
        Fits the model - at the moment not fully implemented

        TODO: implement correctly

        Args:
            observations: Agent ordered, time listed observations per agent
            policy_actions: Agent ordered, time listed policy_actions per agent
            policy_probabilitiess: Agent ordered, time listed policy_probabilitiess per agent
            value_functions: Agent ordered, time listed observalue_functionsvations per agent
            rewards: Agent ordered, time listed rewards per agent
        """

        # 1. Get new policy forward results
        # 2. Calculate GAE
        # 3. Calculate loss (for policy and vf)
        # 4. Do backpropagation and optimizer step

        # Fit the policy network
        policy_action, policy_probabiliy, vf_prediction = self.forward(data)
        print(output)

        # Calculate the loss for the policy network
        policy_loss = self.my_loss(output, y_true)

        # Backpropagate the loss
        policy_loss.backward()

        # Update the policy network
        self.optimizer.step()

    def my_loss(self, output, y_true):
        """
        PPO Loss

        TODO: implement correctly

        Args:
            output:
            y_true:
        """
        # Calculate the loss for the policy network
        policy_loss = torch.nn.functional.cross_entropy(output, y_true)
        policy_loss._requires_grad = True  # pylint: disable=protected-access
        return policy_loss
