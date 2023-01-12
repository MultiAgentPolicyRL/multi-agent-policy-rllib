import os
import copy
import time
import logging
import numpy as np
import tensorflow as tf
import keras.backend as K

from typing import Union

###
from ai_economist_ppo_dt.tensorflow import Actor, Critic
from ai_economist_ppo_dt.utils import get_basic_logger, ms_to_time, time_it
from ai_economist.foundation.base.base_env import BaseEnvironment

### disable eager execution
# tf.compat.v1.disable_eager_execution()


### Choose:
from random import SystemRandom

random = SystemRandom()

# import random


class PPO:
    def __init__(
        self,
        env: BaseEnvironment,
        action_space: int,
        seed: int = None,
        batch_size: int = 32,
        log_level: int = logging.INFO,
        log_path: str = None,
    ) -> None:
        """

        Parameters
        ----------
        env_config : dict
            Dictionary of environment configuration.
        action_space : int
            Number of actions in the environment.
        batch_size : int (default=32)
            Batch size for training.
        """
        if seed is not None:
            random.seed(seed)

        self.logger = get_basic_logger(name="PPO", level=log_level, log_path=log_path)

        self.env = env
        self.action_space = action_space
        self.batch_size = batch_size

        if batch_size > 1000:
            self.logger.warning(
                f"Batch size is very large: {batch_size}. This may cause memory issues (in particular exponential storage time)."
            )

        datetime = log_path.split("/")[-1][:-4]
        self.checkpoint_path = os.path.join(
            os.getcwd(),
            "checkpoints",
            datetime,
        )
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        ### Initialize Actor and Critic networks
        self.actor = Actor(log_level=log_level, log_path=log_path)
        self.critic = Critic(log_level=log_level, log_path=log_path)

        self.logger.info("PPO initialized.")

    # def _generalized_advantage_estimation(values,
    #                                  final_value,
    #                                  discounts,
    #                                  rewards,
    #                                  td_lambda=1.0,
    #                                  time_major=True):
    #     """Computes generalized advantage estimation (GAE).
    #     For theory, see
    #     "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    #     by John Schulman, Philipp Moritz et al.
    #     See https://arxiv.org/abs/1506.02438 for full paper.
    #     Define abbreviations:
    #         (B) batch size representing number of trajectories
    #         (T) number of steps per trajectory
    #     Args:
    #         values: Tensor with shape `[T, B]` representing value estimates.
    #         final_value: Tensor with shape `[B]` representing value estimate at t=T.
    #         discounts: Tensor with shape `[T, B]` representing discounts received by
    #         following the behavior policy.
    #         rewards: Tensor with shape `[T, B]` representing rewards received by
    #         following the behavior policy.
    #         td_lambda: A float32 scalar between [0, 1]. It's used for variance reduction
    #         in temporal difference.
    #         time_major: A boolean indicating whether input tensors are time major.
    #         False means input tensors have shape `[B, T]`.
    #     Returns:
    #         A tensor with shape `[T, B]` representing advantages. Shape is `[B, T]` when
    #         `not time_major`.
    #     """

    #     if not time_major:
    #         with tf.name_scope("to_time_major_tensors"):
    #         discounts = tf.transpose(discounts)
    #         rewards = tf.transpose(rewards)
    #         values = tf.transpose(values)

    #     with tf.name_scope("gae"):

    #         next_values = tf.concat(
    #             [values[1:], tf.expand_dims(final_value, 0)], axis=0)
    #         delta = rewards + discounts * next_values - values
    #         weighted_discounts = discounts * td_lambda

    #         def weighted_cumulative_td_fn(accumulated_td, reversed_weights_td_tuple):
    #         weighted_discount, td = reversed_weights_td_tuple
    #         return td + weighted_discount * accumulated_td

    #         advantages = tf.nest.map_structure(
    #             tf.stop_gradient,
    #             tf.scan(
    #                 fn=weighted_cumulative_td_fn,
    #                 elems=(weighted_discounts, delta),
    #                 initializer=tf.zeros_like(final_value),
    #                 reverse=True))

    #     if not time_major:
    #         with tf.name_scope("to_batch_major_tensors"):
    #         advantages = tf.transpose(advantages)

    #     return tf.stop_gradient(advantages)

    def _get_gaes(
        self,
        rewards: Union[list, np.ndarray],
        values: Union[list, np.ndarray],
        next_values: Union[list, np.ndarray],
        gamma: int = 0.998,
        lamda=0.98,
        normalize=True,
    ) -> np.ndarray:
        """
        Calculate Generalized Advantage Estimation (GAE) for a batch of trajectories.
        ---

        Parameters
        ----------
        rewards : list
            List of rewards for each step in the trajectory.
        values : list
            List of values for each step in the trajectory.
        next_values : list
            List of values for each step in the trajectory.
        gamma : float (default=0.998)
            Discount factor.
        lamda : float (default=0.98)
            GAE parameter.
        normalize : bool (default=True)
            Whether to normalize the GAEs.

        Returns
        -------
        gaes : np.ndarray
            List of GAEs for each step in the trajectory.
        target_values : Tensor
            List of target values for each step in the trajectory.
        """
        deltas = [r + gamma * nv - v for r, nv, v in zip(rewards, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)

        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + gamma * lamda * gaes[t + 1]

        target = gaes + values

        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return np.expand_dims(gaes, -1), tf.convert_to_tensor(
            np.vstack(target), dtype=tf.float32
        )

    # @time_it
    def _act(self, state: np.ndarray) -> np.ndarray:
        """
        Get the one-hot encoded action from Actor network given the state.
        ---
        Parameters
        ---
        state : dict
            State of the environment.

        Returns
        ---
        action : np.ndarray
            action taken by the agent from the random distribution.
        one_hot_action : np.ndarray
            One-hot encoded action.
        prediction : np.ndarray
            Probability distribution over the actions.
        """
        # start_time = time.time()
        # Get the input state as a tensor for the input layer (world-map and flat features)
        input_state = [
            K.expand_dims(state["world-map"], axis=0),
            K.expand_dims(state["flat"], axis=0),
        ]
        # end = time.time()- start_time
        # print(f"Time to expand dims of input state: {end}")
        # Log
        self.logger.debug(
            f"Input state: {(input_state[0].shape[1:], input_state[1].shape[1:])}"
        )

        # Get the prediction from the Actor network
        prediction = np.squeeze(
            self.actor.predict(input_state, workers=20, use_multiprocessing=True)
        )
        # Log
        self.logger.debug(f"Prediction (rounded): {[round(v, 3) for v in prediction]}")

        # Sample an action from the prediction distribution
        action = random.choices(np.arange(self.action_space), weights=prediction)
        # Log
        self.logger.debug(f"Action: {action}")

        # One-hot encode the action
        one_hot_action = np.zeros([self.action_space])
        one_hot_action[action] = 1
        # Log
        self.logger.debug(f"One-hot action: {np.where(one_hot_action == 1)[0]}")

        # Return action, one-hot encoded action, and prediction
        return action, one_hot_action, prediction

    # @time_it
    def get_actions(self, states: np.ndarray) -> np.ndarray:
        """
        Get the one-hot encoded actions from Actor network given the states.
        ---
        Parameters
        ---
        states : np.ndarray
            States of the environment.

        Returns
        ---
        actions : np.ndarray
            actions taken by the agents from the random distribution.
        one_hot_actions : np.ndarray
            One-hot encoded actions.
        predictions : np.ndarray
            Probability distribution over the actions.
        """
        # Initialize dict to store actions
        actions, actions_one_hot, predictions = {}, {}, {}

        # Iterate over agents
        for agent in states.keys():
            if agent != "p":
                # Get the action, one-hot encoded action, and prediction
                action, one_hot_action, prediction = self._act(states[agent])
                # Log
                self.logger.debug(f"Agent {agent} action: {action}")

                # Check before continuing
                if len(action) > 1:
                    self.logger.critical(f"Action is not a scalar: {action}")
                    raise ValueError(f"Action is not a scalar: {action}")

                # Store actions, one-hot encoded actions, and predictions
                actions[agent] = np.array(action)
                actions_one_hot[agent] = one_hot_action
                predictions[agent] = prediction
            # else:
            #     # Store actions, one-hot encoded actions, and predictions
            #     actions[agent] = 0,
            #     actions_one_hot[agent] = 0
            #     predictions[agent] = 0
        # Return actions, one-hot encoded actions, and predictions
        return actions, actions_one_hot, predictions

    def train(
        self,
        states: dict,
        actions: dict,
        rewards: dict,
        predictions: dict,
        next_states: dict,
    ) -> dict:
        """
        Fit Actor and Critic networks. Use it after a batch is collected.
        ---
        Parameters
        ---
        states : list
            List of states in the trajectory.
        actions : list
            List of actions taken in the trajectory.
        rewards : list
            List of rewards in the trajectory.
        predictions : list
            List of predictions from the Actor network in the trajectory.
        next_states : list
            List of next states in the trajectory.
        """
        losses = {
            "0": {"actor": [], "critic": []},
            "1": {"actor": [], "critic": []},
            "2": {"actor": [], "critic": []},
            "3": {"actor": [], "critic": []},
        }

        self.logger.warning(
            "For now removing the 'p' agent from the training. In future THIS MUST BE FIXED."
        )

        for agent in states.keys():
            # Inputs for the Critic network predictions
            input_states = []
            input_next_states = []
            for s, ns in zip(states[agent], next_states[agent]):
                input_states.append(
                    [
                        K.expand_dims(s["world-map"], axis=0),
                        K.expand_dims(s["flat"], axis=0),
                    ]
                )
                input_next_states.append(
                    [
                        K.expand_dims(ns["world-map"], axis=0),
                        K.expand_dims(ns["flat"], axis=0),
                    ]
                )

            self.logger.debug(
                f"Input states: {(input_states[0][0].shape[1:], input_states[0][1].shape[1:])}"
            )
            self.logger.debug(
                f"Input next states: {(input_next_states[0][0].shape[1:], input_next_states[0][1].shape[1:])}"
            )

            # Initialize lists for storing the values and next values
            values = []
            next_values = []
            for i_s, i_ns in zip(input_states, input_next_states):
                values.append(self.critic.predict(i_s))
                next_values.append(self.critic.predict(i_ns))

            # Convert lists to numpy arrays and squeeze (reshape to 1D)
            values = np.squeeze(np.array(values))
            next_values = np.squeeze(np.array(next_values))
            # Log
            self.logger.debug(f"Values: {[round(v, 3) for v in values]}")
            self.logger.debug(f"Next values: {[round(v, 3) for v in next_values]}")

            # Calculate GAEs and target values
            gaes, target_values = self._get_gaes(rewards[agent], values, next_values)
            # Log
            self.logger.debug(f"GAEs: {[round(v[0], 3) for v in gaes]}")
            self.logger.debug(
                f"Target values: {[round(v[0], 3) for v in target_values.numpy()]}"
            )

            # Get y_true for the Actor network
            y_true = tf.convert_to_tensor(
                np.hstack([gaes, predictions[agent], actions[agent]])
            )
            self.logger.debug(f"Actor y_true: {y_true.shape}")

            # Get x values for the Actor&Critic network
            world_map = []
            flat = []
            for s in states[agent]:
                world_map.append(tf.convert_to_tensor(s["world-map"]))
                flat.append(tf.convert_to_tensor(s["flat"]))

            world_map = tf.convert_to_tensor(
                world_map
            )  # Original: tf.convert_to_tensor(world_map)
            flat = tf.convert_to_tensor(flat)  # Original: tf.convert_to_tensor(flat)
            # Log
            self.logger.debug(f"Actor&Critic World map: {world_map.shape}")
            self.logger.debug(f"Actor&Critic Flat: {flat.shape}")

            # Fit the Actor network
            actor_loss_history = self.actor.fit(
                [world_map, flat],
                y_true,
                epochs=1,
                batch_size=2,
                verbose=False,
                shuffle=False,
                workers=20,
            )
            # Log
            self.logger.debug(f"Actor loss history: {actor_loss_history.history}")

            # Get y_true for the Critic network
            # For custom loss function
            y_true = tf.convert_to_tensor(
                np.hstack([target_values, np.expand_dims(values, axis=1)])
            )
            # y_true = tf.convert_to_tensor(values)
            # Log
            # For custom loss function
            self.logger.debug(
                f"Critic y_true: {[(round(y[0], 3), round(y[1], 3)) for y in y_true.numpy()]}"
            )
            # self.logger.debug(f"Critic y_true: {y_true.shape}")

            # Fit the Critic network
            critic_loss_history = self.critic.fit(
                [world_map, flat],
                y_true,
                epochs=1,
                batch_size=2,
                verbose=False,
                shuffle=False,
                workers=20,
            )
            # Log
            self.logger.debug(f"Critic loss history: {critic_loss_history.history}")

            # Log - info
            self.logger.debug(
                f"Actor loss: {actor_loss_history.history['loss'][-1]}, Critic loss: {critic_loss_history.history['loss'][-1]}"
            )

            # global losses
            losses[agent]["actor"] = actor_loss_history.history["loss"][-1]
            losses[agent]["critic"] = critic_loss_history.history["loss"][-1]

        # Should make checkpoint here
        self.checkpoint()

        return losses

    def checkpoint(self):
        """
        Save the weights of the Actor and Critic networks.
        """
        self.actor.save_weights(os.path.join(self.checkpoint_path, "actor.h5"))
        self.critic.save_weights(os.path.join(self.checkpoint_path, "critic.h5"))
        self.logger.info("Checkpoint saved.")

    def populate_batch(self, agents: list = ["0", "1", "2", "3"]) -> dict:
        """
        Populate a batch.

        Returns
        ---
        batch : dict
        """
        base_dict = {agent: [] for agent in agents}

        states_dict = copy.deepcopy(base_dict)
        actions_dict = copy.deepcopy(base_dict)
        rewards_dict = copy.deepcopy(base_dict)
        predictions_dict = copy.deepcopy(base_dict)
        next_states_dict = copy.deepcopy(base_dict)

        self.logger.info(f"Creating a batch of size {self.batch_size}...")
        state = self.env.reset()

        start_timer = time.time()
        for iteration in range(self.batch_size):
            # Get actions, one-hot encoded actions, and predictions
            actions, _, predictions = self.get_actions(state)
            # Log
            self.logger.debug(f"Actions: {actions}")

            # Step the environment with the actions
            next_state, rewards, _, _ = self.env.step(actions)
            # Log
            self.logger.debug(f"Rewards: {rewards}")

            # Append to the batch
            for agent in agents:
                states_dict[agent].append(state[agent])
                actions_dict[agent].append(actions[agent])
                rewards_dict[agent].append(rewards[agent])
                predictions_dict[agent].append(predictions[agent])
                next_states_dict[agent].append(next_state[agent])

            # # Remove Log - info since it's very fast
            # if iteration % 100 == 0 and iteration:
            #     elapsed = ms_to_time((time.time() - start_timer)*1000)
            #     timer = (((time.time() - start_timer)/(iteration+1))*(self.batch_size - iteration -1))*1000
            #     eta = ms_to_time(timer)
            #
            #     self.logger.info(f"Batch creation: {iteration}/{self.batch_size}, Elapsed: {elapsed}, ETA: {eta} [mm:]ss.ms")

            state = copy.deepcopy(next_state)

        self.logger.info(
            f"Batch of size {self.batch_size} created in {ms_to_time((time.time() - start_timer)*1000)}. [mm:]ss.ms"
        )

        r_temp = []
        for agent, values in rewards_dict.items():
            r_temp.append(round((np.count_nonzero(values) / len(values)) * 100, 2))
        self.logger.info(
            f"Rewards neq zero: '0' {r_temp[0]}%, '1' {r_temp[1]}%, '2' {r_temp[2]}%, '3' {r_temp[3]}%"
        )
        del r_temp

        return (
            states_dict,
            actions_dict,
            rewards_dict,
            predictions_dict,
            next_states_dict,
        )

    def test(self, episodes: int = 1) -> None:
        """
        Test the agent.
        """
        self.logger.debug(f"Testing the agent for {episodes} episodes...")
        avg_reward = {"0": [], "1": [], "2": [], "3": []}
        for episode in range(episodes):
            state = self.env.reset()
            done = {"__all__": False}
            step = 0
            while not done.get("__all__") or step < self.batch_size:
                actions, _, _ = self.get_actions(state)
                state, rewards, done, _ = self.env.step(actions)
                for agent in ["0", "1", "2", "3"]:
                    avg_reward[agent].append(rewards[agent])
                step += 1

            self.logger.debug(f"Episode {episode+1}/{episodes} completed.")
            if any([val > 0 for val in rewards.values()]):
                self.logger.debug(
                    f"Average reward: '0' {rewards['0']/step}, '1' {rewards['1']/step}, '2' {rewards['2']/step}, '3' {rewards['3']/step}"
                )

        self.logger.debug(f"Testing completed.")

        return avg_reward


# @deprecated
#
# def actor_ppo_loss(num_actions:int):
#     def loss(y_true, y_pred):
#         # Defined in https://arxiv.org/abs/1707.06347
#         y_true = tf.squeeze(y_true)
#         advantages, prediction_picks, actions = y_true[:1], y_true[1:1+num_actions], y_true[num_actions+1:1 + 2 * num_actions]
#         #print(f"\n\n\n{advantages.shape}, {prediction_picks.shape}, {actions.shape}\n\n\n")
#         LOSS_CLIPPING = 0.2
#         ENTROPY_LOSS = 0.001

#         prob = actions * y_pred
#         old_prob = actions * prediction_picks

#         prob = K.clip(prob, 1e-10, 1.0)
#         old_prob = K.clip(old_prob, 1e-10, 1.0)

#         ratio = K.exp(K.log(prob) - K.log(old_prob))

#         p1 = ratio * advantages
#         p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

#         actor_loss = -K.mean(K.minimum(p1, p2))

#         entropy = -(y_pred * K.log(y_pred + 1e-10))
#         entropy = ENTROPY_LOSS * K.mean(entropy)

#         total_loss = actor_loss - entropy

#         return total_loss
#     return loss

# # def critic_PPO2_loss(values):
# #     def loss(y_true, y_pred):
# #         LOSS_CLIPPING = 0.2
# #         clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
# #         v_loss1 = (y_true - clipped_value_loss) ** 2
# #         v_loss2 = (y_true - y_pred) ** 2

# #         value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
# #         #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
# #         return value_loss
# #     return loss

# def critic_loss(y_true, y_pred):
#     """
#     PPO's loss function, can be with mean or clipped
#     """
#     y_true = tf.squeeze(y_true)

#     values = K.expand_dims(y_true[1], -1)
#     y_true = K.expand_dims(y_true[0], -1)

#     value_loss = K.mean((y_true - y_pred) ** 2)

#     return value_loss
#     # L_CLIP
#     LOSS_CLIPPING = 0.2
#     clipped_value_loss = values + K.clip(
#         y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING
#     )
#     v_loss1 = (y_true - clipped_value_loss) ** 2
#     v_loss2 = (y_true - y_pred) ** 2
#     value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))

#     return value_loss


# # def test_reward(env:BaseEnvironment, model_actor: Model, n_actions: int, limit:int=100) -> dict:
# #     dummy_n = np.zeros((1, n_actions))
# #     dummy_1 = np.zeros((1, 1))

# #     state = env.reset()

# #     done = False
# #     total_reward = {'0': 0, '1': 0, '2': 0, '3': 0}
# #     actions = {'0': 0, '1': 0, '2': 0, '3': 0}

# #     while not done:
# #         for agent_id in total_reward.keys():
# #             input = [
# #                 K.expand_dims(state[agent_id]['world-map'], 0),
# #                 K.expand_dims(state[agent_id]['flat'], 0),
# #                 dummy_n,
# #                 dummy_1,
# #                 dummy_1,
# #                 dummy_1
# #             ]
# #             action_probs = np.reshape(model_actor.predict(input, steps=1, verbose=0), -1)
# #             actions[agent_id] = np.argmax(action_probs)

# #         next_state, reward, done, _ = env.step(actions)
# #         state = next_state

# #         for agent_id in total_reward.keys():
# #             total_reward[agent_id] += reward[agent_id]

# #         limit -= 1

# #         if limit <= 0:
# #             break

# #     return total_reward

# def get_actor_model(n_actions: int=50, conv_filters: tuple = (16, 32), filter_size: int = 3) -> Model:
#     """ Builds the model. Takes in input the parameters that were not specified in the paper. """
#     cnn_in = k.Input(shape=(7, 11, 11))
#     info_input = k.Input(shape=(136,))

#     # oldpolicy_probs = k.Input(shape=(n_actions,))
#     # advantages = k.Input(shape=(1,))
#     # rewards = k.Input(shape=(1,))
#     # values = k.Input(shape=(1,))

#     map_cnn = k.Conv2D(conv_filters[0], filter_size, activation='relu')(cnn_in)
#     map_cnn = k.Conv2D(
#         conv_filters[1], filter_size, activation='relu')(map_cnn)
#     map_cnn = k.Flatten()(map_cnn)

#     mlp1 = k.Concatenate()([map_cnn, info_input])
#     mlp1 = k.Dense(128, activation='relu')(mlp1)
#     mlp1 = k.Dense(128, activation='relu')(mlp1)
#     mlp1 = k.Reshape([1, -1])(mlp1)

#     lstm = k.LSTM(128)(mlp1)
#     mlp2 = k.Dense(n_actions, activation='sigmoid')(lstm)

#     # model = Model(inputs=[cnn_in, info_input, oldpolicy_probs,
#     #               advantages, rewards, values], outputs=mlp2)
#     model = Model(inputs=[cnn_in, info_input], outputs=mlp2)
#     model.compile(optimizer=Adam(lr=0.0003),  # loss='mse')
#                 #   loss=[ppo_loss(
#                 #       oldpolicy_probs=oldpolicy_probs,
#                 #       advantages=advantages,
#                 #       rewards=rewards,
#                 #       values=values
#                 #   )]
#                 loss = actor_ppo_loss(n_actions)
#     )


#     return model

# def get_critic_model() -> Model:
#     cnn_in = k.Input(shape=(7, 11, 11))

#     map_cnn = k.Conv2D(16, 3, activation="relu")(cnn_in)
#     map_cnn = k.Conv2D(32, 3, activation="relu")(map_cnn)
#     map_cnn = k.Flatten()(map_cnn)

#     info_input = k.Input(shape=(136))
#     mlp1 = k.Concatenate()([map_cnn, info_input])
#     mlp1 = k.Dense(128, activation="relu")(mlp1)
#     mlp1 = k.Dense(128, activation="relu")(mlp1)
#     mlp1 = k.Reshape([1, -1])(mlp1)

#     lstm = k.LSTM(128)(mlp1)

#     value = k.Dense(1, name="Out_value_function", activation=None)(lstm)

#     model = Model(inputs=[cnn_in, info_input], outputs = value)
#     model.compile(
#         #loss=[critic_PPO2_loss(old_old_values)],
#         loss = critic_loss,
#         optimizer=Adam(lr=0.0003))

#     return model

#
