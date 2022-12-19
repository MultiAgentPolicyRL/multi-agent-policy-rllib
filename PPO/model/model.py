"""
LSTM NN Model for AI-Economist RL environment
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=unused-import

import copy
import datetime
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from model.model_config import ModelConfig

WORLD_MAP = "world-map"
WORLD_IDX_MAP = "world-idx_map"
ACTION_MASK = "action_mask"

general_logger = logging.getLogger('general')
data_logger = logging.getLogger('data')

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

    # where logit_mask == -10.00.000 softmax puts that probability to 0.

    # Softmax is used to have sum(logit_mask) == 1 -> so it's a probability distibution
    # Addign 1e-3 to avoid zeros
    logit_mask = torch.softmax(logit_mask, dim=1) + 1e-6

    # Makes a Categorical distribution
    dist = torch.distributions.Categorical(probs=logit_mask)
    
    # Gets the action
    action = dist.sample()
    # Gets action log_probability
    # probs = torch.squeeze(dist.log_prob(action)).item()
    return action, dist.probs


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
        self.name = self.model_config.name
        ### This is for managing all the possible inputs without having more networks
        for key, value in self.model_config.observation_space.items():
            ### Check if the input must go through a Convolutional Layer
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

        self.embed_map_idx_policy = nn.Embedding(self.model_config.input_emb_vocab, self.model_config.emb_dim, device=self.model_config.device, dtype=torch.float32)
        self.embed_map_idx_value = nn.Embedding(self.model_config.input_emb_vocab, self.model_config.emb_dim, device=self.model_config.device, dtype=torch.float32)
        self.conv_layers_policy = nn.ModuleList()
        self.conv_layers_value = nn.ModuleList()
        self.conv_shape = (self.conv_shape_r, self.conv_shape_c, self.conv_map_channels + self.conv_idx_channels)

        for i in range(1, self.model_config.num_conv):
            if i == 1:
                self.conv_layers_policy.append(
                    nn.Conv2d(
                        in_channels=self.conv_shape[1],
                        out_channels=self.model_config.filter[0],
                        kernel_size=self.model_config.kernel_size,
                        stride=self.model_config.strides,
                        # padding_mode='same',
                ))
                self.conv_layers_value.append(
                    nn.Conv2d(
                        in_channels=self.conv_shape[1],
                        out_channels=self.model_config.filter[0],
                        kernel_size=self.model_config.kernel_size,
                        stride=self.model_config.strides,
                        # padding_mode='same',
                ))
            self.conv_layers_policy.append(
                    nn.Conv2d(
                        in_channels=self.model_config.filter[0],
                        out_channels=self.model_config.filter[1],
                        kernel_size=self.model_config.kernel_size,
                        stride=self.model_config.strides,
                        # padding_mode='same',
                ))
            self.conv_layers_value.append(
                    nn.Conv2d(
                        in_channels=self.model_config.filter[0],
                        out_channels=self.model_config.filter[1],
                        kernel_size=self.model_config.kernel_size,
                        stride=self.model_config.strides,
                        # padding_mode='same',
                ))
        
        self.conv_dims = self.model_config.kernel_size[0] * self.model_config.strides * self.model_config.filter[1]
        self.flatten_dims = self.conv_dims + self.model_config.observation_space['flat'].shape[0] + len(self.model_config.observation_space['time'])
        self.fc_layer_1_policy = nn.Linear(in_features=self.flatten_dims, out_features=self.model_config.fc_dim)
        self.fc_layer_2_policy = nn.Linear(in_features=self.model_config.fc_dim, out_features=self.model_config.fc_dim)
        self.fc_layer_1_value = nn.Linear(in_features=self.flatten_dims, out_features=self.model_config.fc_dim)
        self.fc_layer_2_value = nn.Linear(in_features=self.model_config.fc_dim, out_features=self.model_config.fc_dim)
        self.lstm_policy = nn.LSTM(
            input_size=self.model_config.fc_dim,
            hidden_size=self.model_config.cell_size,
            num_layers=1,
        )
        self.lstm_value = nn.LSTM(
            input_size=self.model_config.fc_dim,
            hidden_size=self.model_config.cell_size,
            num_layers=1,
        )
        self.layer_norm_policy = nn.LayerNorm(self.model_config.fc_dim)
        self.layer_norm_value = nn.LayerNorm(self.model_config.fc_dim)
        self.output_policy = nn.Linear(in_features=self.model_config.cell_size, out_features=self.model_config.output_size)
        self.output_value = nn.Linear(in_features=self.model_config.cell_size, out_features=1)

        self.relu = nn.ReLU()
        # self.fc_layer_3_policy = nn.Linear(in_features=fc_dim, out_features=output_size)
        # self.fc_layer_3_value = nn.Linear(in_features=fc_dim, out_features=output_size)
        self.softmax = nn.Softmax(dim=1)

        self.hidden_state_h_p = torch.ones(1, self.model_config.cell_size, device=self.model_config.device)
        self.hidden_state_c_p = torch.ones(1, self.model_config.cell_size, device=self.model_config.device)
        self.hidden_state_h_v = torch.ones(1, self.model_config.cell_size, device=self.model_config.device)
        self.hidden_state_c_v = torch.ones(1, self.model_config.cell_size, device=self.model_config.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.model_config.lr)
        
        # Initialize the weights
        for param in self.parameters():
            param.grad = None


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
            _world_map = observation[WORLD_MAP].int()
            _world_idx_map = observation[WORLD_IDX_MAP].int()
            _flat = observation["flat"]
            _time = observation["time"].int()
            _action_mask = observation[ACTION_MASK].int()
        else:
            _world_map = observation[0]
            _world_idx_map = observation[1].long()
            _flat = observation[2]
            _time = observation[3]
            _action_mask = observation[4]

        if self.name == 'p':
            _p0 = observation["p0"]
            _p1 = observation["p1"]
            _p2 = observation["p2"]
            _p3 = observation["p3"]
            
        conv_input_map = torch.permute(_world_map, (0, 2, 3, 1))
        conv_input_idx = torch.permute(_world_idx_map, (0, 2, 3, 1))

        # Concatenate the remainings of the input
        if self.name == 'p':
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

        # Policy
        # Embedd from 100 to 4
        map_embedd = self.embed_map_idx_policy(conv_input_idx)
        # Reshape the map
        map_embedd = torch.reshape(map_embedd, (-1, self.conv_shape_r, self.conv_shape_c, self.conv_idx_channels))
        # Concatenate the map and the idx map
        conv_input = torch.cat([conv_input_map, map_embedd], axis=-1)
        # Convolutional Layers
        for conv_layer in self.conv_layers_policy:
            conv_input = self.relu(conv_layer(conv_input))
        # Flatten the output of the convolutional layers
        flatten = torch.reshape(conv_input, (-1, self.conv_dims)) # 192 is from 32 * 3 * 2
        # Concatenate the convolutional output with the non convolutional input
        fc_in = torch.cat([flatten, non_convolutional_input], axis=-1)
        # Fully Connected Layers
        for i in range(self.model_config.num_fc):
            if i == 0:
                fc_in = self.relu(self.fc_layer_1_policy(fc_in))
            else:
                fc_in = self.relu(self.fc_layer_2_policy(fc_in))
        # Normalize the output
        layer_norm_out = self.layer_norm_policy(fc_in)
        # LSTM
        
        # Project LSTM output to logits 
        lstm_out, hidden = self.lstm_policy(layer_norm_out, (self.hidden_state_h_p, self.hidden_state_c_p))
        self.hidden_state_h_p, self.hidden_state_c_p = hidden[0].detach(), hidden[1].detach()
        if torch.isnan(lstm_out).any():
            general_logger.critical("NAN in lstm_out")
            raise ValueError("NAN in lstm_out")
        # lstm_out = self.output_policy(lstm_out)
        policy_action, policy_probability = apply_logit_mask(self.output_policy(lstm_out), _action_mask)

        #Value
        # Embedd from 100 to 4
        map_embedd = self.embed_map_idx_value(conv_input_idx)
        # Reshape the map
        map_embedd = torch.reshape(map_embedd, (-1, self.conv_shape_r, self.conv_shape_c, self.conv_idx_channels))
        # Concatenate the map and the idx map
        conv_input = torch.cat([conv_input_map, map_embedd], axis=-1)
        # Convolutional Layers
        for conv_layer in self.conv_layers_value:
            conv_input = self.relu(conv_layer(conv_input))
        # Flatten the output of the convolutional layers
        flatten = torch.reshape(conv_input, (-1, self.conv_dims)) # 192 is from 32 * 3 * 2
        # Concatenate the convolutional output with the non convolutional input
        fc_in = torch.cat([flatten, non_convolutional_input], axis=-1)
        # Fully Connected Layers
        for i in range(self.model_config.num_fc):
            if i == 0:
                fc_in = self.relu(self.fc_layer_1_value(fc_in))
            else:
                fc_in = self.relu(self.fc_layer_2_value(fc_in))
        # Normalize the output
        layer_norm_out = self.layer_norm_value(fc_in)
        # LSTM
        
        # Project LSTM output to logits 
        lstm_out, hidden = self.lstm_value(layer_norm_out, (self.hidden_state_h_p, self.hidden_state_c_p))
        self.hidden_state_h_p, self.hidden_state_c_p = hidden[0].detach(), hidden[1].detach()

        if torch.isnan(lstm_out).any():
            logging.critical("NAN in lstm_out")
            raise ValueError("NAN in lstm_out")
        
        value = self.output_value(lstm_out)

        return policy_action, policy_probability, value

    def fit(
        self,
        observations,
        policy_actions,
        policy_probabilities,
        vf_actions,
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

        # GAE Constants:
        gae_gamma = 0.99
        gae_lambda = 0.95
        policy_clip = 0.2
        c1 = 1
        c2 = 0.01

        # 1. Get new policy forward results
        new_policy_action, new_policy_probability, new_vf_action = [],[],[]

        for observation in observations:
            pa, pp, vfp = self.forward(observation)
            new_policy_action.append(pa)
            new_policy_probability.append(pp)
            new_vf_action.append(vfp)

        # 2. Calculate GAE
        deltas = [
            r + gae_gamma * nv - v
            for r, nv, v in zip(rewards, new_vf_action, vf_actions)
        ]
        deltas_len = len(deltas)
        deltas = torch.stack(deltas)

        # By reversing deltas we have it from the nearest to the farthest
        # Ât = δt + (γλ)δt+1 + · · · + · · · + (γλ)T −t+1 δT −1 ,
        for t in reversed(range(deltas_len -1)):
            deltas[t] = deltas[t] + gae_gamma * gae_lambda * deltas[t+1]

        # 2.1 Normalize gaes:
        deltas = (deltas - deltas.mean()) / (deltas.std() + 1e-8)
        advantage = deltas
        # print(advantage)

        # 3. Calculate loss (for policy and vf)
        # prob_ratio = new_policy_probability.exp() / policy_probabilities.exp()
        # Equal to:
        #### NEW METHOD
        # policy_a = []
        # new_policy_a = []
        # for policy, new_policy, action in zip(policy_probabilities, new_policy_probability, policy_actions):
        #     """
        #     We don't have problems with action mask 'nones' because it's always the same, so it's impossibile
        #     having a smth/0 or smth/-inf or smth like this.
        #     """
        #     policy_a.append(torch.squeeze(policy)[action.item()])
        #     new_policy_a.append(torch.squeeze(new_policy)[action.item()])
            
        #     # prob_ratio.append(torch.exp(torch.log()/torch.log()))
        # policy_a = torch.tensor(policy_a)
        # new_policy_a = torch.tensor(new_policy_a)
        
        # prob_ratio = torch.exp(torch.log(new_policy_a)/torch.log(policy_a))
        #### NEW METHOD
        
        
        #### OLD METHOD        
        policy_probabilities = torch.stack(policy_probabilities)
        new_policy_probability = torch.stack(new_policy_probability)
        prob_ratio = new_policy_probability/policy_probabilities
        # prob_ratio = torch.nan_to_num(prob_ratio, 0)
        # print(prob_ratio)
        #### OLD METHOD

        weighted_probs = advantage * prob_ratio
        weighted_clipped_probs = torch.clamp(prob_ratio, 1-policy_clip, 1+policy_clip) * advantage
        policy_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
        # print(policy_loss)

        # ARRIVED HERE
        # sys.exit()
        # VALUE FUNCTION LOSS
        returns = advantage + torch.tensor(vf_actions)
        vf_loss = (returns - torch.tensor(new_vf_action))**2
        vf_loss = vf_loss.mean()

        # ENTROPY
        # entropy = -(y_pred * K.log(y_pred + 1e-10))
        # entropy = ENTROPY_LOSS * K.mean(entropy)
        entropy = -(new_policy_probability * torch.log(new_policy_probability + 1e-10))
        entropy = torch.mean(entropy)

        # TOTAL LOSS
        total_loss = policy_loss + c1*vf_loss - c2*entropy

        data_logger.info(f"total_loss,{total_loss}")
      
        # 4. Do backpropagation and optimizer step
        # total_loss=total_loss.detach()
        total_loss.backward()
        self.optimizer.step()
