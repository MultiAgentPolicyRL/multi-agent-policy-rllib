import os
import copy
import time
import logging
import numpy as np
import tensorflow as tf
import keras.backend as K

from typing import Union

###
from ai_economist_ppo_dt.tensorflow.models import LSTMModel
from ai_economist_ppo_dt.utils import get_basic_logger, ms_to_time, time_it
from ai_economist.foundation.base.base_env import BaseEnvironment


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
        initial_state = env.reset()
        self.action_dim = {"a": 1, "p": env.all_agents[4].action_spaces.shape[0]}
        self.action_space = {
            "a": env.all_agents[0].action_spaces,
            "p": np.sum(env.all_agents[4].action_spaces),
        }
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
        self.agent = LSTMModel(
            initial_state["0"],
            name="AgentLSTM",
            output_size=self.action_space.get("a"),
            log_level=log_level,
            log_path=log_path,
        )
        # This row is added to start the @tf.function decorator and compile the function
        self.agent.predict(self.agent.prepare_inputs(initial_state["0"]))

        self.planner = LSTMModel(
            initial_state["p"],
            name="PlannerLSTM",
            output_size=self.action_space.get("p"),
            log_level=log_level,
            log_path=log_path,
        )
        # This row is added to start the @tf.function decorator and compile the function
        self.planner.predict(self.planner.prepare_inputs(initial_state["p"]))

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

    # @time_it
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
    def _act(
        self, state: dict, agent: str = "a", hidden_states: dict = None
    ) -> np.ndarray:
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


        # Get the prediction from the Actor network
        if agent == 'a':
            # Prediction -> (1, 1, 50) -> contains the probability distribution over the actions
            # Values -> (1, 1, 1) -> contains the value estimate for the state
            # state_in_h_p -> (1, 128) -> contains the hidden state of the planner
            # state_in_c_p -> (1, 128) -> contains the hidden state of the planner
            # state_in_h_v -> (1, 128) -> contains the hidden state of the planner
            # state_in_c_v -> (1, 128) -> contains the hidden state of the planner
            prediction, values, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v = self.agent.predict(input_state)
        else:
            # Prediction -> (1, 1, 154) -> contains the probability distribution over the actions
            # Values -> (1, 1, 1) -> contains the value estimate for the state
            # state_in_h_p -> (1, 128) -> contains the hidden state of the planner
            # state_in_c_p -> (1, 128) -> contains the hidden state of the planner
            # state_in_h_v -> (1, 128) -> contains the hidden state of the planner
            # state_in_c_v -> (1, 128) -> contains the hidden state of the planner
            prediction, values, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v = self.planner.predict(input_state)

        """
        # Get the input state as a tensor for the input layer
        if agent == "a":
            input_state = self.agent.prepare_inputs(state, hidden_states, 1)
        else:
            input_state = self.planner.prepare_inputs(state, hidden_states, 1)
        # Log
        self.logger.debug(f"Input state: {[x.shape for x in input_state.values()]}")

        # Get the prediction from the Actor network
        if agent == "a":
            (
                prediction,
                values,
                state_in_h_p,
                state_in_c_p,
                state_in_h_v,
                state_in_c_v,
            ) = self.agent.predict(input_state)
        else:
            (
                prediction,
                values,
                state_in_h_p,
                state_in_c_p,
                state_in_h_v,
                state_in_c_v,
            ) = self.planner.predict(input_state)

        # Remove shape of (1, 1, 50) to (50,)
        squeezed_prediction = np.squeeze(prediction)
        # Log
        self.logger.debug(
            f"Prediction (rounded): {[round(v, 3) for v in squeezed_prediction]}"
        )

        if hidden_states is None:
            hidden_states = {
                "state_in_h_p": None,
                "state_in_c_p": None,
                "state_in_h_v": None,
                "state_in_c_v": None,
            }

        # Append the additional states to the dictionary
        hidden_states["state_in_h_p"] = state_in_h_p
        hidden_states["state_in_c_p"] = state_in_c_p
        hidden_states["state_in_h_v"] = state_in_h_v
        hidden_states["state_in_c_v"] = state_in_c_v

        # Sample an action from the prediction distribution
        if agent == "a":
            action = random.choices(
                np.arange(self.action_space[agent]),
                weights=squeezed_prediction,
                k=self.action_dim[agent],
            )
        else:
            action = []
            _actions_single_dim = self.action_space[agent] // self.action_dim[agent]
            _possible_actions = np.arange(_actions_single_dim)
            for i in range(self.action_dim[agent]):
                _weights = squeezed_prediction[
                    _actions_single_dim * i : _actions_single_dim * (i + 1)
                ]
                action.append(
                    random.choices(_possible_actions, weights=_weights, k=1)[0]
                )
        # Log
        self.logger.debug(f"Action: {action}")

        # One-hot encode the action
        one_hot_action = np.zeros([self.action_space[agent]])
        if agent == "a":
            one_hot_action[action] = 1
        else:
            for i in range(self.action_dim[agent]):
                one_hot_action[action[i]] = 1
        # Log
        self.logger.debug(f"One-hot action: {np.where(one_hot_action == 1)}")

        # Return action, one-hot encoded action, prediction and additional states
        dict_to_return = {
            "world-map": input_state["world-map"],
            "world-idx_map": input_state["world-idx_map"],
            "time": input_state["time"],
            "flat": input_state["flat"],
            "action_mask": input_state["action_mask"],
        }
        return action, one_hot_action, prediction, dict_to_return, values, hidden_states

    # @time_it
    def get_actions(
        self, states: np.ndarray, additional_states: dict = None
    ) -> np.ndarray:
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
        (
            actions,
            actions_one_hot,
            predictions,
            new_input_state,
            values,
            hidden_states,
        ) = ({}, {}, {}, {}, {}, {})

        # Iterate over agents
        for agent in states.keys():

            # Get the action, one-hot encoded action, and prediction
            action, one_hot_action, prediction, n_i_s, v, h_s = self._act(
                states[agent], "a" if agent != "p" else "p", additional_states
            )
            # Log
            self.logger.debug(f"Agent {agent} action: {action}")

            # Check before continuing
            # if len(action) > 1:
            #     self.logger.critical(f"Action is not a scalar: {action}")
            #     raise ValueError(f"Action is not a scalar: {action}")

            # Store actions, one-hot encoded actions, and predictions
            actions[agent] = np.array(action)
            actions_one_hot[agent] = one_hot_action
            predictions[agent] = prediction
            hidden_states[agent] = h_s
            new_input_state[agent] = n_i_s
            values[agent] = v

        # Return actions, one-hot encoded actions, predictions and hidden states
        return (
            actions,
            actions_one_hot,
            predictions,
            new_input_state,
            values,
            hidden_states,
        )

    def train(
        self,
        states: dict,
        actions: dict,
        rewards: dict,
        predictions: dict,
        next_states: dict,
        values: dict,
        hidden_states: dict,
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
            "p": {"actor": [], "critic": []},
        }

        self.logger.warning(
            "For now removing the 'p' agent from the training. In future THIS MUST BE FIXED."
        )

        hidden_states = {
            "a": {
                "state_in_h_p": [],
                "state_in_c_p": [],
                "state_in_h_v": [],
                "state_in_c_v": [],
            },
            "p": {
                "state_in_h_p": [],
                "state_in_c_p": [],
                "state_in_h_v": [],
                "state_in_c_v": [],
            },
        }

        for agent in states.keys():
            # Inputs for the Critic network predictions
            input_states = copy.deepcopy(states[agent])
            # input_next_states = copy.deepcopy(next_states[agent])
            # for s, ns in zip(states[agent], next_states[agent]):
            #     input_states.append([
            #         K.expand_dims(s['world-map'], axis=0),
            #         K.expand_dims(s['flat'], axis=0)
            #     ])
            #     input_next_states.append([
            #         K.expand_dims(ns['world-map'], axis=0),
            #         K.expand_dims(ns['flat'], axis=0)
            #     ])

            # self.logger.debug(f"Input states: {len(input_states)}")
            # self.logger.debug(f"Input next states: {len(input_next_states)}")

            # # Initialize lists for storing the values and next values
            # values = []
            # next_values = []
            # for i_s, i_ns in zip(input_states, input_next_states):
            #     if agent == 'p':
            #         i_s = self.critic_planner.prepare_inputs(i_s)
            #         i_ns = self.critic_planner.prepare_inputs(i_ns)
            #         values.append(self.critic_planner.predict(i_s))
            #         next_values.append(self.critic_planner.predict(i_ns))
            #     else:
            #         i_s = self.critic_agent.prepare_inputs(i_s)
            #         i_ns = self.critic_agent.prepare_inputs(i_ns)
            #         values.append(self.critic_agent.predict(i_s))
            #         next_values.append(self.critic_agent.predict(i_ns))

            # Convert lists to numpy arrays and squeeze (reshape to 1D)
            # values = np.squeeze(np.array(values))
            # next_values = np.squeeze(np.array(next_values))
            # Log
            # self.logger.debug(f"Values: {[round(v, 3) for v in values]}")
            # self.logger.debug(f"Next values: {[round(v, 3) for v in next_values]}")

            # Calculate GAEs and target values
            ### FIXME: values and next_values should be of shape 10 but one the next of the other
            ### FIXME: To solve it temporarily values will be = [0] + values[:-1], next_values is the original `values`
            pred_values = [tf.convert_to_tensor(0.0)] + [
                tf.squeeze(v) for v in values[agent][:-1]
            ]
            actual_values = [tf.squeeze(v) for v in values[agent]]
            gaes, target_values = self._get_gaes(
                rewards[agent], pred_values, actual_values
            )
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

            # Log
            # self.logger.debug(f"Actor&Critic World map: {world_map.shape}")
            # self.logger.debug(f"Actor&Critic Flat: {flat.shape}")

            # Fit the Actor network
            # FIXME: ADD LOSS FUNCTION
            exit()
            actor_loss_history = self.actor.fit(
                input_states,
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

    # @time_it
    def _convert_states_to_tensors(self, states: list) -> list:
        """
        Convert states to tensors.

        Parameters
        ---
        states : list
            List of states.

        Returns
        ---
        states : list
            List of states.
        """
        for agent, state in states.items():
            if agent != "p":
                states[agent] = self.agent.prepare_inputs(
                    state, append_hidden_states=False
                )
            else:
                states[agent] = self.planner.prepare_inputs(
                    state, append_hidden_states=False
                )

        return states

    # @time_it
    def populate_batch(self, agents: list = ["0", "1", "2", "3", "p"]) -> dict:
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
        values_dict = copy.deepcopy(base_dict)
        hidden_states_dict = copy.deepcopy(base_dict)

        self.logger.info(f"Creating a batch of size {self.batch_size}...")
        state = self._convert_states_to_tensors(self.env.reset())

        start_timer = time.time()
        for iteration in range(self.batch_size):
            # Get actions, one-hot encoded actions, and predictions
            actions, _, predictions, new_input_state, values, h_s = self.get_actions(
                state, hidden_states_dict[agent][-1] if iteration > 0 else None
            )
            # Log
            self.logger.debug(f"Actions: {actions}")

            # Step the environment with the actions
            next_state, rewards, _, _ = self.env.step(actions)
            next_state = self._convert_states_to_tensors(next_state)
            # Log
            self.logger.debug(f"Rewards: {rewards}")

            # Append to the batch
            for agent in agents:
                states_dict[agent].append(new_input_state[agent])
                actions_dict[agent].append(
                    K.expand_dims(K.expand_dims(actions[agent], axis=0), axis=0)
                )
                rewards_dict[agent].append(rewards[agent])
                predictions_dict[agent].append(predictions[agent])
                next_states_dict[agent].append(next_state[agent])
                values_dict[agent].append(values[agent])
                hidden_states_dict[agent].append(h_s[agent])

            state = copy.deepcopy(next_state)

            # if iteration:
            #     exit()
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
            values_dict,
            hidden_states_dict,
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
