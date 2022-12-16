import os
import sys
import torch
import logging
import torch.nn as nn
from typing import Dict, List, Tuple, Union

from ai_economist_ppo_dt.utils import get_basic_logger, time_it

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
    logit_mask = torch.ones(logits.shape)# * -10000000
    logit_mask = logit_mask * (1 - mask)
    logit_mask = logits + logit_mask

    ## Softmax is used to have sum(logit_mask) == 1 -> so it's a probability distibution
    logit_mask = torch.softmax(logit_mask, dim=1)
    ## Makes a Categorical distribution
    dist = torch.distributions.Categorical(logit_mask)
    # Gets the action
    action = dist.sample()
    # Gets action log_probability
    # probs = torch.squeeze(dist.log_prob(action)).item()

    return action, logit_mask

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
        self.logger = get_basic_logger(name, level=log_level, log_path=log_path)
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

        self.embed_map_idx_policy = nn.Embedding(input_emb_vocab, emb_dim, device=device, dtype=torch.float32)
        self.embed_map_idx_value = nn.Embedding(input_emb_vocab, emb_dim, device=device, dtype=torch.float32)
        self.conv_layers_policy = nn.ModuleList()
        self.conv_layers_value = nn.ModuleList()
        self.conv_shape = (self.conv_shape_r, self.conv_shape_c, self.conv_map_channels + self.conv_idx_channels)

        for i in range(1, self.num_conv):
            if i == 1:
                self.conv_layers_policy.append(
                    nn.Conv2d(
                        in_channels=self.conv_shape[1],
                        out_channels=filter[0],
                        kernel_size=kernel_size,
                        stride=strides,
                        # padding_mode='same',
                ))
                self.conv_layers_value.append(
                    nn.Conv2d(
                        in_channels=self.conv_shape[1],
                        out_channels=filter[0],
                        kernel_size=kernel_size,
                        stride=strides,
                        # padding_mode='same',
                ))
            self.conv_layers_policy.append(
                    nn.Conv2d(
                        in_channels=filter[0],
                        out_channels=filter[1],
                        kernel_size=kernel_size,
                        stride=strides,
                        # padding_mode='same',
                ))
            self.conv_layers_value.append(
                    nn.Conv2d(
                        in_channels=filter[0],
                        out_channels=filter[1],
                        kernel_size=kernel_size,
                        stride=strides,
                        # padding_mode='same',
                ))
        
        self.conv_dims = kernel_size[0] * strides * filter[1]
        self.flatten_dims = self.conv_dims + obs['flat'].shape[0] + len(obs['time'])
        self.fc_layer_1_policy = nn.Linear(in_features=self.flatten_dims, out_features=fc_dim)
        self.fc_layer_2_policy = nn.Linear(in_features=fc_dim, out_features=fc_dim)
        self.fc_layer_1_value = nn.Linear(in_features=self.flatten_dims, out_features=fc_dim)
        self.fc_layer_2_value = nn.Linear(in_features=fc_dim, out_features=fc_dim)
        self.lstm_policy = nn.LSTM(
            input_size=fc_dim,
            hidden_size=cell_size,
            num_layers=1,
        )
        self.lstm_value = nn.LSTM(
            input_size=fc_dim,
            hidden_size=cell_size,
            num_layers=1,
        )
        self.layer_norm_policy = nn.LayerNorm(fc_dim)
        self.layer_norm_value = nn.LayerNorm(fc_dim)
        self.output_policy = nn.Linear(in_features=cell_size, out_features=output_size)
        self.output_value = nn.Linear(in_features=cell_size, out_features=1)

        self.relu = nn.ReLU()
        # self.fc_layer_3_policy = nn.Linear(in_features=fc_dim, out_features=output_size)
        # self.fc_layer_3_value = nn.Linear(in_features=fc_dim, out_features=output_size)
        self.softmax = nn.Softmax(dim=1)

        self.hidden_state_h_p = torch.ones(1, self.cell_size, device=self.device)
        self.hidden_state_c_p = torch.ones(1, self.cell_size, device=self.device)
        self.hidden_state_h_v = torch.ones(1, self.cell_size, device=self.device)
        self.hidden_state_c_v = torch.ones(1, self.cell_size, device=self.device)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)

        # Initialize the weights
        for param in self.parameters():
            param.grad = None

        if log_level == logging.DEBUG:
            from torchinfo import summary
            input_array = [[
                torch.IntTensor(obs['world-map']).unsqueeze(0).to(self.device),
                torch.IntTensor(obs['world-idx_map']).unsqueeze(0).to(self.device),
                torch.FloatTensor(obs['flat']).unsqueeze(0).to(self.device),
                torch.FloatTensor(obs['time']).unsqueeze(0).to(self.device),
                torch.IntTensor(obs['action_mask']).unsqueeze(0).to(self.device),
            ]]
            summary(self, input_data=input_array, depth=25, verbose=1)
            exit()
        
        self.logger.info("Model created successfully")

    def forward(self, x: dict, training: bool = False):
        if isinstance(x, dict):
            _world_map = x[WORLD_MAP].int()
            _world_idx_map = x[WORLD_IDX_MAP].int()
            _flat = x["flat"]
            _time = x["time"].int()
            _action_mask = x[ACTION_MASK].int()
        else:
            _world_map = x[0]
            _world_idx_map = x[1].long()
            _flat = x[2]
            _time = x[3]
            _action_mask = x[4]

        if self.name == 'p':
            _p0 = x["p0"]
            _p1 = x["p1"]
            _p2 = x["p2"]
            _p3 = x["p3"]
            
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
        for i in range(self.num_fc):
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
        lstm_out = self.output_policy(lstm_out)
        action, logits = apply_logit_mask(lstm_out, _action_mask)    

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
        for i in range(self.num_fc):
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
        value = self.output_value(lstm_out)
    

        # OLD CODE -> TO DELETE -> The problem with this code was that the weights are shared between the policy and the value, while they should not be
        # for tag in ['_policy', '_value']:
        #     # Embedd from 100 to 4
        #     map_embedd = self.embed_map_idx(conv_input_idx)
        #     # Reshape the map
        #     map_embedd = torch.reshape(map_embedd, (-1, self.conv_shape_r, self.conv_shape_c, self.conv_idx_channels))
        #     # Concatenate the map and the idx map
        #     conv_input = torch.cat([conv_input_map, map_embedd], axis=-1)
        #     # Convolutional Layers
        #     for conv_layer in self.conv_layers:
        #         conv_input = self.relu(conv_layer(conv_input))
        #     # Flatten the output of the convolutional layers
        #     flatten = torch.reshape(conv_input, (-1, self.conv_dims)) # 192 is from 32 * 3 * 2
        #     # Concatenate the convolutional output with the non convolutional input
        #     fc_in = torch.cat([flatten, non_convolutional_input], axis=-1)
        #     # Fully Connected Layers
        #     for i in range(self.num_fc):
        #         if i == 0:
        #             fc_in = self.relu(self.fc_layer_1(fc_in))
        #         else:
        #             fc_in = self.relu(self.fc_layer_2(fc_in))
        #     # Normalize the output
        #     layer_norm_out = self.layer_norm(fc_in)
        #     # LSTM
            
        #     # Project LSTM output to logits or value
        #     #
        #     if tag == '_policy':
        #         lstm_out, hidden = self.lstm_policy(layer_norm_out, (self.hidden_state_h_p, self.hidden_state_c_p))
        #         self.hidden_state_h_p, self.hidden_state_c_p = hidden[0].detach(), hidden[1].detach()
        #         if torch.isnan(lstm_out).any():
        #             self.logger.critical("NAN in lstm_out")
        #             raise ValueError("NAN in lstm_out")
        #         lstm_out = self.fc_layer_3(lstm_out)
        #         logits = self.softmax(lstm_out)#apply_logit_mask(self.output_policy(lstm_out), _action_mask)
        #     else:
        #         lstm_out, hidden = self.lstm_value(layer_norm_out, (self.hidden_state_h_v, self.hidden_state_c_v))
        #         self.hidden_state_h_v, self.hidden_state_c_v = hidden[0].detach(), hidden[1].detach()
        #         value = self.output_value(lstm_out)

        return action, logits, value
        #return logits, value

    def fit(self, states: List[dict], epochs: int, batch_size: int, gaes: List[torch.FloatTensor], predictions: List[torch.FloatTensor], actions: List[torch.FloatTensor], verbose: Union[bool, int] = 0) -> torch.Tensor:
        """
        Function to fit the model.
        """
        if self.logger.level == logging.DEBUG or verbose:
            torch.autograd.set_detect_anomaly(True)
            if verbose:
                temp_level = self.logger.level
                self.logger.setLevel(logging.DEBUG) 

        losses = []

        for epoch in range(epochs):
            _text = f"Epoch {epoch+1}/{epochs}"
            self.logger.debug(f"{_text:-^20}")
            for batch in range(batch_size):
                # Get the predictions
                action, logits, values = self.forward(states[batch])

                if torch.isnan(logits).any():
                    self.logger.error(f"s:\n{states[batch]}")
                    self.logger.error(f"Logits: {logits}")
                    exit()
                    
                # Log
                self.logger.debug(f"Outputs: ")
                self.logger.debug(f"Logits: {logits}")
                self.logger.debug(f"Values: {values}")

                # Calculate the loss
                loss = self.custom_loss(logits, values, gaes[batch], predictions[batch], actions[batch])
                # Log
                self.logger.debug(f"Loss: {loss}")

                loss.backward()
                self.optimizer.step()

                if self.logger.level == logging.DEBUG and batch+1 % 100 == 0:
                    self.logger.debug(f"Batch {batch+1}/{batch_size}")
                    self.logger.debug(f"Loss: {loss}")

                losses.append(loss.item())

        if verbose:
            self.logger.setLevel(temp_level)

        return losses

    def custom_loss(self, out_logits, out_value, gaes, predictions, actions):
        """
        Custom loss from [arxiv:1707.06347](https://arxiv.org/abs/1707.06347).
        """
        # Constants
        _epsilon = 0.2
        _entropy = 0.001

        prob = actions * out_logits
        old_prob = actions * predictions
        
        prob = torch.clamp(prob, 1e-10, 1.0)
        old_prob = torch.clamp(old_prob, 1e-10, 1.0)

        ratio = torch.exp(torch.log(prob) - torch.log(old_prob))

        p1 = ratio * gaes
        p2 = torch.clamp(ratio, 1.0 - _epsilon, 1.0 + _epsilon) * gaes

        policy_loss = -torch.min(p1, p2).mean()
        value_loss = 0.5 * (out_value - gaes).pow(2).mean()

        # Calculate the entropy without considering `nan` values
        entropy = -torch.nansum(out_logits * torch.log(out_logits + 1e-10), dim=1).mean()

        loss = policy_loss + _entropy * entropy
        
        return loss

    
    def save(self, path: str):
        """
        Function to save the model.
        """
        if not path.endswith('.pth'):
            path += 'checkpoint.pth'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """
        Function to load the model.
        """
        self.load_state_dict(torch.load(path))
                