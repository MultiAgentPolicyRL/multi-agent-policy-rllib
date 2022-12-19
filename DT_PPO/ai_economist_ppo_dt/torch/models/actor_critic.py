import os
import sys
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
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
    logit_mask = torch.ones(logits.shape) * -10000000
    logit_mask = logit_mask * (1 - mask)
    logit_mask = logits + logit_mask

    ##Softmax is used to have sum(logit_mask) == 1 -> so it's a probability distibution
    logit_mask = torch.softmax(logit_mask, dim=1) # * 1e3
    
    # Creates a torch distribution with weights from logit_mask (logit_mask is a probability distribution)
    dist = torch.distributions.Categorical(logit_mask)

    # Sample an action from the distribution
    action = dist.sample()

    # Get the log_probability of the action
    # log_prob = dist.log_prob(action)

    return action, logit_mask

class LSTMModel(nn.Module):
    """
    Actor&Critic (Policy) Model.
    =====
    
    

    """
    
    def __init__(self, obs: dict, name: str, emb_dim: int = 4, cell_size: int = 128, input_emb_vocab: int = 100, num_conv: int = 2, fc_dim: int = 128, num_fc: int = 2, filter: Tuple[int, int]= (16, 32), kernel_size: Tuple[int, int] = (3, 3), strides: int = 2, output_size: int = 50, lr: float = 0.0003, entropy: float = 0.001, epsilon: float = 0.2, log_level: int = logging.INFO, log_path: str = None, device: str = 'cpu') -> None:
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
        self.lr = 0.01 # lr
        # self.weight_decay = 0.01
        self.momentum = 0.9
        self.output_size = output_size
        self.device = device

        self._epsilon = epsilon
        self._entropy = entropy

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
        self.softmax = nn.Softmax()

        # self.hidden_state_h_p = torch.ones(1, self.cell_size, device=self.device)
        # self.hidden_state_c_p = torch.ones(1, self.cell_size, device=self.device)
        # self.hidden_state_h_v = torch.ones(1, self.cell_size, device=self.device)
        # self.hidden_state_c_v = torch.ones(1, self.cell_size, device=self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)#, weight_decay=self.weight_decay)
        # self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

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
            sys.exit()
        
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
        lstm_out, hidden = self.lstm_policy(layer_norm_out)#, (self.hidden_state_h_p, self.hidden_state_c_p))
        self.hidden_state_h_p, self.hidden_state_c_p = hidden[0].detach(), hidden[1].detach()
        lstm_out = self.output_policy(lstm_out)
        # Check that 'lstm_out' is not NaN
        if torch.isnan(lstm_out).any():
            self.logger.error(f"'lstm_out' is NaN\n{lstm_out}")
            sys.exit()
        # Mask the logits
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
        lstm_out, hidden = self.lstm_value(layer_norm_out)#, (self.hidden_state_h_p, self.hidden_state_c_p))
        self.hidden_state_h_p, self.hidden_state_c_p = hidden[0].detach(), hidden[1].detach()
        value = self.output_value(lstm_out)

        if torch.isnan(action.any()) or torch.isnan(logits.any()) or torch.isnan(value.any()):
            if torch.isnan(action.any()):
                self.logger.critical(f"Action contains NaNs:")
                self.logger.critical(f"{action}")
            elif torch.isnan(logits.any()):
                self.logger.critical(f"Logits contains NaNs:")
                self.logger.critical(f"{logits}")
            elif torch.isnan(value.any()):
                self.logger.critical(f"Value contains NaNs:")
                self.logger.critical(f"{value}")
            sys.exit(-1)

        return action, logits, value

    def get_minibatches(self, states: List[dict], gaes: List[torch.FloatTensor], predictions: List[torch.FloatTensor], actions: List[torch.FloatTensor], rewards: List[torch.FloatTensor], mini_batch_size: int, shuffle: bool = True):
        """
        Build a list of minibatches from the states.
        """
        state_size = len(states)
        new_states = []
        for i in range(state_size//mini_batch_size):
            _world_map = Variable(torch.stack([state['world-map'].squeeze(0) for state in states[i * mini_batch_size : (i + 1) * mini_batch_size]]))
            _world_idx_map = Variable(torch.stack([state['world-idx_map'].squeeze(0) for state in states[i * mini_batch_size : (i + 1) * mini_batch_size]]))
            _flat = Variable(torch.stack([state['flat'].squeeze(0) for state in states[i * mini_batch_size : (i + 1) * mini_batch_size]]))
            _time = Variable(torch.stack([state['time'].squeeze(0) for state in states[i * mini_batch_size : (i + 1) * mini_batch_size]]))
            _action_mask = Variable(torch.stack([state['action_mask'].squeeze(0) for state in states[i * mini_batch_size : (i + 1) * mini_batch_size]]))

            _gae = Variable(torch.stack([gae for gae in gaes[i * mini_batch_size : (i + 1) * mini_batch_size]]))
            _predictions = Variable(torch.stack([prediction for prediction in predictions[i * mini_batch_size : (i + 1) * mini_batch_size]]))
            _actions = Variable(torch.stack([action for action in actions[i * mini_batch_size : (i + 1) * mini_batch_size]]))
            _rewards = Variable(torch.stack([reward for reward in rewards[i * mini_batch_size : (i + 1) * mini_batch_size]]))
            
            x = {
                'world-map': _world_map,
                'world-idx_map': _world_idx_map,
                'flat': _flat,
                'time': _time,
                'action_mask': _action_mask,
                'gaes': _gae,
                'predictions': _predictions,
                'actions': _actions,
                'rewards': _rewards
            }
            new_states.append(x)
        
        if shuffle:
            random.shuffle(new_states)

        return new_states

    def fit(self, states: List[dict], gaes: List[torch.FloatTensor], predictions: List[torch.FloatTensor], actions: List[torch.FloatTensor], rewards: List[torch.FloatTensor], epochs: int, batch_size: int, verbose: Union[bool, int] = 0) -> torch.Tensor:
        """
        Function to fit the model.
        """
        if self.logger.level == logging.DEBUG or verbose:
            torch.autograd.set_detect_anomaly(True)
            if verbose:
                temp_level = self.logger.level
                self.logger.setLevel(logging.DEBUG) 

        losses = []
        mini_batches = self.get_minibatches(states, gaes, predictions, actions, rewards, batch_size//10, shuffle=True)

        bar = tqdm(range(epochs), desc="Epoch", disable=not verbose)
        for epoch in bar:
            bar.set_description(f"Epoch {epoch+1}/{epochs}")
            mini_batches_iter = copy.deepcopy(mini_batches)
            for batch in mini_batches_iter:
                gae = batch.pop('gaes')
                action = batch.pop('actions')
                prediction = batch.pop('predictions')
                reward = batch.pop('rewards')

                # Get the logits and value
                _out_action, _out_logits, _out_value = self.forward(batch)
                # Log
                self.logger.debug(f"Action: {_out_action}")
                self.logger.debug(f"Logits: {_out_logits}")
                self.logger.debug(f"Value: {_out_value}")

                # Calculate the loss
                loss = self.custom_loss(_out_logits, _out_value, gae, prediction, reward)
                losses.append(loss.item())
                # Log
                self.logger.debug(f"Loss: {loss}")

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                bar.set_postfix(loss=loss.item())

        if verbose:
            self.logger.setLevel(temp_level)

        return losses

    def custom_loss(self, out_logits: torch.FloatTensor, out_value: torch.FloatTensor, gaes: torch.FloatTensor, predictions: torch.FloatTensor, returns: torch.FloatTensor) -> torch.Tensor:
        r"""
        Custom loss function for PPO [arxiv:1707.06347](https://arxiv.org/abs/1707.06347).
        """
        # ### OLD
        # # Constants
        # _epsilon = 0.2
        # _entropy = 0.001

        # prob = returns * out_logits
        # old_prob = returns * predictions
        
        # prob = torch.clamp(prob, 1e-10, 1.0)
        # old_prob = torch.clamp(old_prob, 1e-10, 1.0)

        # ratio = torch.exp(torch.log(prob) - torch.log(old_prob))

        # p1 = ratio * gaes
        # p2 = torch.clamp(ratio, 1.0 - _epsilon, 1.0 + _epsilon) * gaes

        # policy_loss = -torch.min(p1, p2).mean()
        # value_loss = torch.mean(torch.square(out_value - returns)) 
        # # value_loss = 0.5 * (out_value - gaes).pow(2).mean()

        # # Calculate the entropy without considering `nan` values
        # entropy = -torch.nansum(out_logits * torch.log(out_logits + 1e-10), dim=1).mean()

        # loss = policy_loss + value_loss + _entropy * entropy
        
        # return loss
        def actor_loss():
            r"""Computes the actor loss for Proximal Policy Optimization (PPO).
            
            The actor loss is used to update the policy in PPO. It is based on the difference
            between the old policy and the new policy, as well as the advantage value for each
            action. The loss is calculated using the surrogate loss function, which is defined as:
            
                `L = min(r_t * A, clip(r_t, 1-e, 1+e) * A)`
                
            where r_t is the ratio of the new to the old policy, A is the advantage value, and
            e is the clip ratio. The loss is then averaged across all actions and states.
            
            Args:
                advantages: A tensor of shape (batch_size, num_actions) representing the
                    advantage value for each action.
                old_log_probs: A tensor of shape (batch_size, num_actions) representing the
                    log probability of the old policy for each action.
                log_probs: A tensor of shape (batch_size, num_actions) representing the
                    log probability of the new policy for each action.
                clip_ratio: A scalar value representing the clip ratio (e in the equation above).
                
            Returns:
                A scalar value representing the actor loss.
            """
            # Calculate the ratio of the new to the old policy
            # r_t = torch.exp(torch.sub(out_logits, predictions))
            # r_t = torch.exp(torch.sub(torch.log(out_logits), torch.log(predictions)))
            # r_t = torch.divide(out_logits, predictions)
            r_t = []
            _out_logits = out_logits.detach().numpy()
            _predictions = predictions.detach().numpy().squeeze(1)
            for i in range(_out_logits.shape[0]):
                for j in range(_out_logits.shape[1]):
                    if _predictions[i][j] == 0 and _out_logits[i][j] == 0:
                        r_t.append(1.0)
                    else:
                        r_t.append(_predictions[i][j] / _out_logits[i][j])
            r_t = torch.tensor(r_t).view(out_logits.shape)
            # r_t = torch.nan_to_num(r_t, nan=0.0, posinf=0.0, neginf=0.0)

            # Calculate the surrogate loss
            surr_loss = torch.minimum(r_t * gaes, 
                                torch.clamp(r_t, 1-self._epsilon, 1+self._epsilon) * gaes)
            
            # Return the mean surrogate loss
            return torch.mean(-surr_loss)

        def critic_loss():
            r"""Computes the critic loss for Proximal Policy Optimization (PPO).
            
            The critic loss is used to update the value function in PPO. It is based on the
            difference between the predicted value and the actual return for each state. The
            loss is calculated using the mean squared error (MSE) between the predicted and
            actual returns, and is averaged across all states.
            
            Args:
                values: A tensor of shape (batch_size, 1) representing the predicted value
                    for each state.
                returns: A tensor of shape (batch_size, 1) representing the actual return
                    for each state.
                advantages: A tensor of shape (batch_size, num_actions) representing the
                    advantage value for each action.
                
            Returns:
                A scalar value representing the critic loss.
            """
            # Calculate the value loss
            value_loss = torch.mean(torch.square(out_value - returns)) # 0.5 * (out_value - returns).pow(2).mean()

            # Return the critic loss
            return value_loss
        
        # Calculate the actor loss
        loss_actor = actor_loss()

        # Calculate the critic loss
        loss_critic = critic_loss()

        # Calculate the entropy loss
        loss_entropy = torch.mean(-out_logits * torch.exp(out_logits))

        # Calculate the total loss
        loss = loss_actor + loss_critic - self._entropy * loss_entropy

        # Check for NaNs
        if torch.isnan(loss):
            self.logger.error(f"Loss is NaN: {round(loss.item(),3)}")
            self.logger.error(f"Actor loss: {round(loss_actor.item(),3)}")
            self.logger.error(f"Critic loss: {round(loss_critic.item(),3)}")
            self.logger.error(f"Entropy loss: {round(loss_entropy.item(),3)}")
            sys.exit(1)

        # Return the total loss
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
                