import os
import torch
import logging
import numpy as np
import torch.nn as nn

from typing import Optional, Tuple, Union


WORLD_MAP = "world-map"
WORLD_IDX_MAP = "world-idx_map"
ACTION_MASK = "action_mask"


def apply_logit_mask(logits, mask):
    """Mask values of 1 are valid actions."
    " Add huge negative values to logits with 0 mask values."""
    logit_mask = torch.ones(logits.shape) * -10000000
    logit_mask = logit_mask * (1 - mask)

    return logits + logit_mask

class LSTMModel(nn.Module):
    """
    Actor&Critic (Policy) Model.
    =====
    
    

    """
    
    def __init__(self, obs: dict, name: str, emb_dim: int = 4, cell_size: int = 128, input_emb_vocab: int = 100, num_conv: int = 2, fc_dim: int = 128, num_fc: int = 2, filter: Tuple[int, int]= (16, 32), kernel_size: Tuple[int, int] = (3, 3), strides: int = 2, output_size: int = 50, lr: float = 0.0003, log_level: int = logging.INFO, log_path: str = None, device: str = 'cpu') -> None:
        """
        Initialize the ActorCritic Model.
        """
        super(LSTMModel, self).__init__()

        self.name = name
        # self.logger = get_basic_logger(name, level=log_level, log_path=log_path)
        self.shapes = dict()

        ### Initialize some variables needed here
        self.cell_size = cell_size
        self.num_outputs = output_size
        self.input_emb_vocab = input_emb_vocab
        self.emb_dim = emb_dim
        self.num_conv = num_conv
        self.fc_dim = fc_dim
        self.num_fc = num_fc
        self.filter = filter
        self.kernel_size = kernel_size
        self.strides = strides
        self.lr = lr
        self.output_size = output_size
        self.device = device

        ### This is for managing all the possible inputs without having more networks
        for key, value in obs.items():
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
                self.conv_idx_channels = value.shape[0] * emb_dim
        ###

        self.embed_map_idx = nn.Embedding(input_emb_vocab, emb_dim, device=device, dtype=torch.float32)
        self.conv_layers = nn.ModuleList()
        self.conv_shape = (self.conv_shape_r, self.conv_shape_c, self.conv_map_channels + self.conv_idx_channels)

        for i in range(1, self.num_conv):
            if i == 1:
                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=self.conv_shape[1],
                        out_channels=filter[0],
                        kernel_size=kernel_size,
                        stride=strides,
                        # padding_mode='same',
                ))
            self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=filter[0],
                        out_channels=filter[1],
                        kernel_size=kernel_size,
                        stride=strides,
                        # padding_mode='same',
                ))
        
        self.conv_dims = kernel_size[0] * strides * filter[1]
        self.flatten_dims = self.conv_dims + obs['flat'].shape[0] + len(obs['time'])
        self.fc_layer_1 = nn.Linear(in_features=self.flatten_dims, out_features=fc_dim)
        self.fc_layer_2 = nn.Linear(in_features=fc_dim, out_features=fc_dim)
        self.lstm = nn.LSTM(
            input_size=fc_dim,
            hidden_size=cell_size,
            num_layers=1,
        )
        self.layer_norm = nn.LayerNorm(fc_dim)
        self.output_policy = nn.Linear(in_features=cell_size, out_features=output_size)
        self.output_value = nn.Linear(in_features=cell_size, out_features=1)

        self.relu = nn.ReLU()

        self.hidden_state_h_p = torch.zeros(1, self.cell_size, device=self.device)
        self.hidden_state_c_p = torch.zeros(1, self.cell_size, device=self.device)
        self.hidden_state_h_v = torch.zeros(1, self.cell_size, device=self.device)
        self.hidden_state_c_v = torch.zeros(1, self.cell_size, device=self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # self.logger.info("Model created successfully")

    # @time_it
    def forward(self, input: dict):
        if isinstance(input, dict):
            _world_map = input[WORLD_MAP]
            _world_idx_map = input[WORLD_IDX_MAP]
            _flat = input["flat"]
            _time = input["time"]
            _action_mask = input[ACTION_MASK]
        else:
            _world_map = input[0]
            _world_idx_map = input[1].long()
            _flat = input[2]
            _time = input[3]
            _action_mask = input[4]

        if self.name == 'p':
            _p0 = input["p0"]
            _p1 = input["p1"]
            _p2 = input["p2"]
            _p3 = input["p3"]

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
        
        for tag in ['_policy', '_value']:
            # Embedd from 100 to 4
            map_embedd = self.embed_map_idx(conv_input_idx) # TO CHECK WHICH IS THE INPUT -- DONE
            # Reshape the map
            map_embedd = torch.reshape(map_embedd, (-1, self.conv_shape_r, self.conv_shape_c, self.conv_idx_channels))
            # Concatenate the map and the idx map
            conv_input = torch.cat([conv_input_map, map_embedd], axis=-1)
            # Convolutional Layers
            for conv_layer in self.conv_layers:
                conv_input = self.relu(conv_layer(conv_input))
            # Flatten the output of the convolutional layers
            flatten = torch.reshape(conv_input, (-1, self.conv_dims)) # 192 is from 32 * 3 * 2
            # Concatenate the convolutional output with the non convolutional input
            fc_in = torch.cat([flatten, non_convolutional_input], axis=-1)
            # Fully Connected Layers
            for i in range(self.num_fc):
                if i == 0:
                    fc_in = self.relu(self.fc_layer_1(fc_in))
                else:
                    fc_in = self.relu(self.fc_layer_2(fc_in))
            # Normalize the output
            layer_norm_out = self.layer_norm(fc_in)
            # LSTM
            
            # Project LSTM output to logits or value
            #
            if tag == '_policy':
                lstm_out, hidden = self.lstm(layer_norm_out, (self.hidden_state_h_p, self.hidden_state_c_p))
                self.hidden_state_h_p, self.hidden_state_c_p = hidden
                logits = apply_logit_mask(self.output_policy(lstm_out), _action_mask)
            else:
                lstm_out, hidden = self.lstm(layer_norm_out, (self.hidden_state_h_v, self.hidden_state_c_v))
                self.hidden_state_h_v, self.hidden_state_c_v = hidden
                value = self.output_value(lstm_out)
        
        return logits, value


    def fit(self, input, y_true):
        # Fit the Actor network
        output = self.forward(input)
        
        # Calculate the loss for the Actor network
        actor_loss = self.my_loss(output, y_true)

        # Backpropagate the loss
        actor_loss.backward()

        # Update the Actor network
        self.optimizer.step()
    
    def my_loss(self, output, y_true):
        # Calculate the loss for the Actor network
        actor_loss = torch.nn.functional.cross_entropy(output, y_true)
        actor_loss._requires_grad = True
        return actor_loss
                