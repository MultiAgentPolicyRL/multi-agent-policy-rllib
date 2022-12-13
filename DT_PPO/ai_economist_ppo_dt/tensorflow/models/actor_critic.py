import os
import gym
import logging
import numpy as np
import tensorflow as tf
# import keras.layers as k
import keras.backend as K
from typing import Optional, Tuple, Union

####
from keras.optimizers import Adam
from keras.models import Model, load_model

from ai_economist_ppo_dt.utils import get_basic_logger, time_it

WORLD_MAP = "world-map"
WORLD_IDX_MAP = "world-idx_map"
ACTION_MASK = "action_mask"


def apply_logit_mask(logits, mask):
    """Mask values of 1 are valid actions."
    " Add huge negative values to logits with 0 mask values."""
    logit_mask = tf.ones_like(logits) * -10000000
    logit_mask = logit_mask * (1 - mask)

    return logits + logit_mask

class LSTMModel:
    """
    Actor&Critic (Policy) Model.
    =====
    
    

    """
    # @tf.function
    def __init__(self, obs: dict, name: str, emb_dim: int = 4, cell_size: int = 128, input_emb_vocab: int = 100, num_conv: int = 2, fc_dim: int = 128, num_fc: int = 2, filter: Tuple[int, int]= (16, 32), kernel_size: Tuple[int, int] = (3, 3), strides: int = 2, output_size: int = 50, lr: float = 0.0003, log_level: int = logging.INFO, log_path: str = None) -> None:
        """
        Initialize the ActorCritic Model.
        """
        self.logger = get_basic_logger(name, level=log_level, log_path=log_path)
        self.shapes = dict()

        ### Initialize some variables needed here
        self.cell_size = cell_size
        self.num_outputs = output_size
        input_dict = {}
        non_convolutional_input_list = []
        # logits, values, state_h_p, state_c_p, state_h_v, state_c_v = (None,None,None,None,None,None,)
        ###

        ### This is for managing all the possible inputs without having more networks
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                # Original
                shape = (None,) + value.shape
                self.shapes[key] = value.shape
            else:
                # Original
                shape = (None,) + (len(value),)
                self.shapes[key] = (len(value),)
            input_dict[key] = tf.keras.layers.Input(shape=shape, name=key)

            ### Check if the input must go through a Convolutional Layer
            if key == ACTION_MASK:
                pass
            elif key == WORLD_MAP:
                conv_shape_r, conv_shape_c, conv_map_channels = (
                    value.shape[1],
                    value.shape[2],
                    value.shape[0],
                )
            elif key == WORLD_IDX_MAP:
                conv_idx_channels = value.shape[0] * emb_dim
            else:
                non_convolutional_input_list.append(key)
        ###

        ### Convolutional Layers
        conv_shape = (
            conv_shape_r,
            conv_shape_c,
            conv_map_channels + conv_idx_channels,
        )
        conv_input_map = tf.keras.layers.Permute((1, 3, 4, 2))(input_dict[WORLD_MAP])
        conv_input_idx = tf.keras.layers.Permute((1, 3, 4, 2))(input_dict[WORLD_IDX_MAP])
        ###

        ### Concatenate all the non convolutional inputs
        non_convolutional_input = tf.keras.layers.concatenate([input_dict[key] for key in non_convolutional_input_list])#, axis=-1)
        ###

        ### Define the cell states and the hidden states
        state_in_h_p = tf.keras.layers.Input(shape=(cell_size,), name="h_pol")
        state_in_c_p = tf.keras.layers.Input(shape=(cell_size,), name="c_pol")
        state_in_h_v = tf.keras.layers.Input(shape=(cell_size,), name="h_val")
        state_in_c_v = tf.keras.layers.Input(shape=(cell_size,), name="c_val")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in")
        ###

        ### Define the policy and value function models
        ### Two possibilities: '_policy' and '_value'
        ### The '_policy' model is used to sample actions
        ### The '_value' model is used to estimate the value of a state
        for tag in ['_policy', '_value']:
            ### Define the embedding layer that embedd from 100 to 4
            map_embedding = tf.keras.layers.Embedding(
                input_dim=input_emb_vocab,
                output_dim=emb_dim,
                name="map_embedding"+tag,
            )
            ###
            ### Reshape world-idx_map from (*,11,11,2) to (2,11,11,8) 
            conv_idx_embedding = tf.keras.layers.Reshape(
                (-1, conv_shape_r, conv_shape_c, conv_idx_channels)
                )(map_embedding(conv_input_idx))
            ###
            ### Concatenate the convolutional inputs
            conv_input = tf.keras.layers.concatenate(
                [conv_input_map, conv_idx_embedding], axis=-1
            )
            ###
            ### Define the convolutional module
            conv_module = tf.keras.models.Sequential(name="conv_module"+tag)
            ###
            ### Add the convolutional layers
            conv_module.add(
                tf.keras.layers.Conv2D(
                    filters=filter[0],
                    kernel_size=kernel_size,
                    strides=strides,
                    activation="relu",
                    input_shape=conv_shape,
                    name="conv2D_0"+tag,
                )
            )
            for i in range(num_conv - 1):
                conv_module.add(
                    tf.keras.layers.Conv2D(
                        filters=filter[1],
                        kernel_size=kernel_size,
                        strides=strides,
                        activation="relu",
                        name=f"conv2D_{i+1}{tag}",
                    )
                )
            ###
            ### Flatten the output of the convolutional module
            conv_module.add(tf.keras.layers.Flatten(name="flatten"+tag))
            ###
            ### Apply a layer to every temporal slice of an input
            conv_td = tf.keras.layers.TimeDistributed(conv_module)(conv_input)
            ###
            ### Concatenate the convolutional and non convolutional inputs
            dense = tf.keras.layers.concatenate([conv_td, non_convolutional_input])
            ###
            ### Add FC layers to the LSTM
            for layer in range(num_fc):
                new_layer = tf.keras.layers.Dense(
                    fc_dim, activation="relu", name=f"fc_{layer}{tag}"
                )
                dense = new_layer(dense)
            ###
            ### Define the LSTM states based on the tag
            if tag == "_policy":
                state_in = [state_in_h_p, state_in_c_p]
            else: #"_value"
                state_in = [state_in_h_v, state_in_c_v]
            ###
            ### Normalize the output of the FC layers
            dense = tf.keras.layers.LayerNormalization(name="layer_norm"+tag)(dense)
            ###
            ### Define the LSTM output and the LSTM states
            lstm_out, state_h, state_c = tf.keras.layers.LSTM(
                cell_size, return_sequences=True, return_state=True, name="lstm"+tag
            )(inputs=dense, mask=tf.sequence_mask(seq_in), initial_state=state_in)
            ###
            ### Project LSTM output to logits or value
            output = tf.keras.layers.Dense(
                output_size if tag == "_policy" else 1,
                activation=tf.keras.activations.linear,
                name="logits" if tag == "_policy" else "value",
            )(lstm_out)
            ###
            ### Define the output based on the tag
            if tag == "_policy":
                state_h_p, state_c_p = state_h, state_c
                logits = apply_logit_mask(output, input_dict[ACTION_MASK])
            else: #"_value"
                state_h_v, state_c_v = state_h, state_c
                values = output
            ### 
        ### Define the model 
        self.model = tf.keras.Model(
            inputs=[layer for layer in input_dict.values()] + [seq_in, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v],
            outputs=[logits, values, state_h_p, state_c_p, state_h_v, state_c_v],
        )
        ###
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        if log_level == logging.DEBUG:
            self.model.summary()
        
        self.logger.info("Model created successfully")

    def custom_loss(self, ):
        def loss(y_true, y_pred):
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
        return loss

    # def _loss_wrapper(self, num_actions:int) -> tf.Tensor:
    #     """
    #     Wrapper for the loss function.

    #     Parameters
    #     ----------
    #     num_actions : int
    #         The number of actions.
        
    #     Returns
    #     -------
    #     loss : tf.Tensor
    #         The loss function.
    #     """
    #     @tf.function
    #     def loss(y_true, y_pred) -> tf.Tensor: 
    #         """
    #         Loss function for the actor (policy) model. 
    #         -----
    #         Defined in [arxiv:1707.06347](https://arxiv.org/abs/1707.06347).

    #         Parameters
    #         -----
    #         y_true : tf.Tensor
    #             The true action.
    #         y_pred : tf.Tensor
    #             The predicted action.
            
    #         Returns
    #         -----
    #         loss : tf.Tensor
    #             The loss.
    #         """
    #         with tf.device(self.device):
    #             y_true = tf.squeeze(y_true)

    #             advantages, prediction_picks, actions = y_true[:,:1], y_true[:, 1:1+num_actions], y_true[:, -1:]
                
    #             LOSS_CLIPPING = 0.2
    #             ENTROPY_LOSS = 0.001

    #             prob = actions * y_pred
    #             old_prob = actions * prediction_picks

    #             prob = K.clip(prob, 1e-10, 1.0)
    #             old_prob = K.clip(old_prob, 1e-10, 1.0)

    #             ratio = K.exp(K.log(prob) - K.log(old_prob))
                
    #             p1 = ratio * advantages
    #             p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

    #             actor_loss = -K.mean(K.minimum(p1, p2))
                
    #             entropy = -(y_pred * K.log(y_pred + 1e-10))
    #             entropy = ENTROPY_LOSS * K.mean(entropy)
                
    #             total_loss = actor_loss - entropy

    #         return total_loss 
    #     return loss

    # @time_it
    @tf.function
    def predict(self, inputs: Union[dict, list] = None):
        if isinstance(inputs, dict):
            inputs = list(inputs.values())

        if self.logger.level == logging.DEBUG:
            for x in inputs:
                self.logger.debug(f"Input shape: {x.shape}")

        return self.model(inputs)

    def fit(self, inputs, output):
        return self.model.fit(inputs, output)

    @tf.function
    def initial_state(self, seq_in: Union[bool, int] = True):
        if seq_in:
            return [
                tf.ones((1, 1), dtype=tf.float32),                  # seq_in
                tf.ones((1, self.cell_size), dtype=tf.float32),    # state_in_h_p
                tf.ones((1, self.cell_size), dtype=tf.float32),    # state_in_c_p
                tf.ones((1, self.cell_size), dtype=tf.float32),    # state_in_h_v
                tf.ones((1, self.cell_size), dtype=tf.float32),    # state_in_c_v
            ]
        
        return [
            tf.zeros((1, self.cell_size), dtype=tf.float32),
            tf.zeros((1, self.cell_size), dtype=tf.float32),
            tf.zeros((1, self.cell_size), dtype=tf.float32),
            tf.zeros((1, self.cell_size), dtype=tf.float32),
        ]
    
    # @tf.function
    def prepare_inputs(self, state: Union[np.ndarray, list, dict], hidden_states: dict = None, seq_in: int = 1, append_hidden_states: bool = True):
        """
        Function to prepare data to be processed by the model.

        --- Structure of the input ---
        [world-map, world-idx_map, flat, time, action_mask, seq_in, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v]
        """
        input_state = dict()

        for key, value in state.items():
            input_state[key] = self.correct_dimension(key, value)

        if append_hidden_states:
            if hidden_states is not None:
                input_state['seq_in'] = tf.ones((1, 1), dtype=tf.float32)
                for key, value in hidden_states.items():
                    input_state[key] = value
            else:
                input_state['seq_in'] = tf.ones((1, 1), dtype=tf.float32)
                for hid in ['state_in_h_p', 'state_in_c_p', 'state_in_h_v', 'state_in_c_v']:
                    input_state[hid] = tf.zeros((1, self.cell_size), dtype=tf.float32)

        return input_state

    def correct_dimension(self, key, value):
        if isinstance(value, list):
            value = np.array(value)
        if value.shape == self.shapes[key]:
            for _ in range(2):
                value = K.expand_dims(value, axis=0)
        
        return value