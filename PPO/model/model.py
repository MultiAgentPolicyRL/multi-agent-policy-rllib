"""
tf.kears actor-critic model
"""
import logging
import sys

# from gym.spaces import MultiDiscrete
# from tensorflow import keras as k
import keras as k
import numpy as np
import tensorflow as tf
from model.model_config import ModelConfig
from functools import wraps
import time

# mirrored_strategy = tf.distribute.MirroredStrategy()

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # logging.debug(f"Function {func.__name__} Took {total_time:.4f} seconds")
        logging.debug(f"{total_time}")
        return result

    return timeit_wrapper


def dict_to_tensor_dict(a_dict: dict):
    """
    pass a single agent obs, returns it's tensor_dict
    """
    tensor_dict = {}
    for key, value in a_dict.items():
        tensor_dict[key] = tf.convert_to_tensor(value, name=key)
        tensor_dict[key] = tf.expand_dims(tensor_dict[key], axis=0)

    return tensor_dict


class ActorModel(object):
    """
    Network's Actor model
    """

    def __init__(self, model_config: ModelConfig) -> tf.keras.Model:
        """
        Builds the model.
        """
        self.action_space = model_config.action_space

        with tf.device("CPU:0"):
        # with mirrored_strategy.scope():    
            self.cnn_in = tf.keras.Input(shape=(7, 11, 11))
            self.map_cnn = tf.keras.layers.Conv2D(16, 3, activation="relu")(self.cnn_in)
            self.map_cnn = tf.keras.layers.Conv2D(32, 3, activation="relu")(self.map_cnn)
            self.map_cnn = tf.keras.layers.Flatten()(self.map_cnn)

            self.info_input = tf.keras.Input(shape=(136))
            self.mlp1 = tf.keras.layers.Concatenate()([self.map_cnn, self.info_input])
            self.mlp1 = tf.keras.layers.Dense(128, activation="relu")(self.mlp1)
            self.mlp1 = tf.keras.layers.Dense(128, activation="relu")(self.mlp1)
            self.mlp1 = tf.keras.layers.Reshape([1, -1])(self.mlp1)

            self.lstm = tf.keras.layers.LSTM(128, unroll=True)(self.mlp1)

            # Policy pi - needs to be a probabiliy value
            self.action_probs = tf.keras.layers.Dense(
                self.action_space, name="Out_probs_actions", activation="softmax"
            )(self.lstm)

            self.actor: tf.keras.Model = tf.keras.Model(
                inputs=[self.cnn_in, self.info_input], outputs=self.action_probs
            )

            # reason of Adam optimizer lr=0.0003 https://github.com/ray-project/ray/issues/8091
            self.actor.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                loss=self.ppo_loss,
                run_eagerly=False,
            )

        logging.critical(self.actor.summary())
    
    @tf.function
    def ppo_loss(self, y_true, y_pred):
        """
        Defined in https://arxiv.org/abs/1707.06347
        """
        # advantages - \hat{A_t} (ex: from gae)
        # predictions, actions_one_hot
        advantages, prediction_picks, actions = (
            y_true[:, :1],
            y_true[:, 1 : 1 + self.action_space],
            y_true[:, 1 + self.action_space :],
        )

        EPSYLON = 0.2
        ENTROPY_LOSS = 0.001


        # pi = actions_one_hot [0,50] * actions_prediction_distribution
        # pi_old = actions_one_hot [0,50] * old_actions_prediction_distribution
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        #### Probably done for optimization:
        #### https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        prob = tf.keras.backend.clip(prob, 1e-10, 1.0)
        old_prob = tf.keras.backend.clip(old_prob, 1e-10, 1.0)

        ratio = tf.keras.backend.exp(tf.keras.backend.log(prob) - tf.keras.backend.log(old_prob))
        ####


        # probabiliy_ratio*advantage_estimation
        # rt(θ) ∗ ˆAt
        p1 = ratio * advantages
        
        # CLIP: clip(probabiliy_ratio, 1-Epsylon, 1+Epsylon)*advantage_estimation)
        # clip(rt(θ), 1 − ε, 1 + ε) ∗ ˆAt)
        p2 = (
            tf.keras.backend.clip(
                ratio, min_value=1 - EPSYLON, max_value=1 + EPSYLON
            )
            * advantages
        )

        # L_CLIP: min(probabiliy_ratio*advantage_estimation, clip(probabiliy_ratio, 1-Epsylon, 1+Epsylon)*advantage_estimation)
        # LCLIP (θ) = ˆEt[min(rt(θ) ∗ ˆAt, clip(rt(θ), 1 − ε, 1 + ε) ∗ ˆAt)] 
        actor_loss = -tf.keras.backend.mean(tf.keras.backend.minimum(p1, p2))

        # critic_loss = # TODO

        entropy = -(y_pred * tf.keras.backend.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * tf.keras.backend.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    # @timeit
    @tf.function
    def predict(self, obs):
        """
        If you remove the reshape good luck finding that softmax sum != 1.
        """ 
        action = tf.squeeze(self.actor(
                [
                    
                    tf.expand_dims(obs["world-map"], 0),
                    tf.expand_dims(obs["flat"], 0),
                ],
            ))
        # return tf.divide(action,tf.reduce_sum(action))
        return action



class CriticModel(object):
    """
    Network's Critic model
    """

    def __init__(self, model_config: ModelConfig) -> tf.keras.Model:
        """Builds the model. Takes in input the parameters that were not specified in the paper."""

        with tf.device("CPU:0"):
        # with mirrored_strategy.scope():
            cnn_in = tf.keras.Input(shape=(7, 11, 11))
            map_cnn = k.layers.Conv2D(16, 3, activation="relu")(cnn_in)
            map_cnn = k.layers.Conv2D(32, 3, activation="relu")(map_cnn)
            map_cnn = k.layers.Flatten()(map_cnn)

            info_input = k.Input(shape=(136))
            mlp1 = k.layers.Concatenate()([map_cnn, info_input])
            mlp1 = k.layers.Dense(128, activation="relu")(mlp1)
            mlp1 = k.layers.Dense(128, activation="relu")(mlp1)
            mlp1 = k.layers.Reshape([1, -1])(mlp1)

            lstm = k.layers.LSTM(128, unroll=True)(mlp1)
            # None or tanh, DO NOT USE SOFTMAX!
            value_pred = k.layers.Dense(1, name="Out_value_function", activation=None)(
                lstm
            )

            self.critic: k.Model = k.Model(
                inputs=[cnn_in, info_input], outputs=value_pred
            )

            # reason of Adam optimizer https://github.com/ray-project/ray/issues/8091
            # 0.0003
            self.critic.compile(
                optimizer=k.optimizers.Adam(learning_rate=0.0003),
                loss=self.loss,
                run_eagerly=False,
            )

    @tf.function
    def loss(self, y_true, y_pred):
        """
        PPO's loss function, can be with mean or clipped
        """
        value_loss = k.backend.mean((y_true - y_pred) ** 2)

        # L_CLIP
        # LOSS_CLIPPING = 0.2
        # clipped_value_loss = values + k.backend.clip(
        #     y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING
        # )
        # v_loss1 = (y_true - clipped_value_loss) ** 2
        # v_loss2 = (y_true - y_pred) ** 2
        # value_loss = 0.5 * k.backend.mean(k.backend.maximum(v_loss1, v_loss2))
        return value_loss

    @tf.function
    def batch_predict(self, obs: list):
        """
        Calculates a batch of prediction for n_obs
        """
        return self.critic([tf.convert_to_tensor(obs['world-map']), tf.convert_to_tensor(obs['flat'])])
