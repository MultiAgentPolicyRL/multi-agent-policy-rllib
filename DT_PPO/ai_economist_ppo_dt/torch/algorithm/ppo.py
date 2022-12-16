import os
import copy
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Union, Dict, Tuple

###
from ai_economist_ppo_dt.torch import LSTMModel
from ai_economist_ppo_dt.utils import get_basic_logger, ms_to_time, time_it
from ai_economist.foundation.base.base_env import BaseEnvironment


### Choose:
from random import SystemRandom
random = SystemRandom()

#import random

class PPO():
    def __init__(self, env: BaseEnvironment, action_space: int, seed: int = None, epochs: int = 1, batch_size: int = 32, device: str = 'cpu', log_level: int = logging.INFO, log_path: str = None) -> None:
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
        initial_state = self.env.reset()
        self.action_space = action_space
        self.batch_size = batch_size
        self.epochs = epochs

        if batch_size > 1000:
            self.logger.warning(f"Batch size is very large: {batch_size}. This may cause memory issues (in particular exponential storage time).")

        datetime = log_path.split("/")[-1][:-4]
        self.checkpoint_path = os.path.join(os.getcwd(), "checkpoints", datetime,)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.action_dim = {'a': 1, 'p': env.all_agents[4].action_spaces.shape[0]}
        self.action_space = {'a': env.all_agents[0].action_spaces, 'p': np.sum(env.all_agents[4].action_spaces)}

        ### Initialize Actor and Critic networks
        self.device = device
        self.actor = LSTMModel(initial_state['0'], name="AgentLSTM", output_size=self.action_space.get('a'), log_level=log_level, log_path=log_path, device=self.device).to(self.device)
        self.planner = LSTMModel(initial_state['0'], name="PlannerLSTM", output_size=self.action_space.get('p'), log_level=log_level, log_path=log_path, device=self.device).to(self.device)
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

    @torch.no_grad()
    def _get_gaes(self, rewards: List[torch.FloatTensor], values: List[torch.FloatTensor], next_values: List[torch.FloatTensor], gamma:float=0.998, lamda:float=0.98, normalize:bool=True,) -> np.ndarray:
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
        # deltas = torch.stack(deltas)
        gaes = copy.deepcopy(deltas)

        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + gamma * lamda * gaes[t + 1]

        target = gaes + values

        gaes = torch.stack(gaes).to(self.device)

        if normalize:
            gaes = (gaes - torch.mean(gaes)) / (torch.std(gaes) + 1e-8)

        return [x for x in gaes], target

    # @time_it
    @torch.no_grad()
    def _act(self, state: List[torch.FloatTensor], agent: str = 'a') -> np.ndarray:
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
        if agent == 'a':
            # Log
            self.logger.debug(f"Input state: {(x.shape[1:] for x in state)}")
            # Get the prediction from the Actor network
            #with torch.no_grad():
            logits, value = self.actor(state)
            _prediction = torch.squeeze(logits).detach().numpy()
            # Log the prediction
            self.logger.debug(f"Prediction: {[round(float(p), 3) for p in _prediction]}")

            # Sample an action from the prediction distribution
            action = torch.FloatTensor(random.choices(np.arange(self.action_space[agent]), weights=_prediction)).to(self.device)
            # Log
            self.logger.debug(f"Action: {action}")

            # One-hot encode the action
            one_hot_action = torch.zeros([self.action_space[agent]]).to(self.device)
            one_hot_action[int(action.item())] = 1
            # Log
            self.logger.debug(f"One-hot action: {np.where(one_hot_action == 1)[0]}")
        else:
            action = torch.IntTensor([random.randint(0,21) for _ in range(7)])
            one_hot_action = torch.zeros([self.action_space[agent]]).to(self.device)
            for action_idx in action:
                one_hot_action[action_idx.item()] = 1
            logits = torch.zeros([self.action_space[agent]]).to(self.device)
            value = torch.zeros([1]).to(self.device)
        
        return action, one_hot_action, logits, value

    # @time_it
    @torch.no_grad()
    def get_actions(self, states: Dict[str, list]) -> np.ndarray:
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
        actions, actions_one_hot, predictions, values = {}, {}, {}, {}

        # Iterate over agents
        for agent in states.keys():
            # Get the action, one-hot encoded action, and prediction
            action, one_hot_action, prediction, value = self._act(states[agent], 'a' if agent != 'p' else 'p')
            # Log
            self.logger.debug(f"Agent {agent} action: {action}")
            
            # Store actions, one-hot encoded actions, and predictions
            actions[agent] = action
            actions_one_hot[agent] = one_hot_action
            predictions[agent] = prediction
            values[agent] = value
                
        return actions, actions_one_hot, predictions, values

    @torch.no_grad()
    def convert_state_to_tensor(self, state: Dict[str, list]) -> Dict[str, list]:
        """
        Convert a state to a tensor.
        
        Parameters
        ---
        state : dict
            The state to convert.
        
        Returns
        ---
        tensor : list
            The converted state.
        """
        new_state = {'0': None, '1': None, '2': None, '3': None, 'p': None}

        for agent in state.keys():
            temp_state = {}
            for key, value in state[agent].items():
                if key in ['flat', 'time', 'p0', 'p1', 'p2', 'p3']:
                    temp_state[key] = torch.FloatTensor(value).unsqueeze(0).to(self.device)
                else:
                    temp_state[key] = torch.IntTensor(value).unsqueeze(0).to(self.device)
            new_state[agent] = temp_state
            
        
        return new_state
    
    # @time_it
    @torch.no_grad()
    def populate_batch(self, agents: list = ['0', '1', '2', '3', 'p']) -> dict:
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

        # Append first values for the critic network
        # This is done because when computing the gaes we need the value of the next state
        # We can't get the value of the next state because we don't have the next state if 
        # we consider only the `self.batch_size` steps
        # So we append the first value of the next state to the values list in order to obtain in training:
        # `values = [0, value_1, value_2, ..., value_{n-1}]` and
        # `next_values = [value_1, value_2, ..., value_n]`

        for key in values_dict.keys():
            values_dict[key].append(torch.zeros(1, 1).to(self.device))

        self.logger.info(f"Creating a batch of size {self.batch_size}...")
        state = self.env.reset()
        state = self.convert_state_to_tensor(state)
        
        start_timer = time.time()
        for iteration in range(self.batch_size):
            # Get actions, one-hot encoded actions, and predictions
            actions, _, predictions, values = self.get_actions(state)
            # Log
            self.logger.debug(f"Actions: {actions}")

            # Step the environment with the actions
            step_actions = {agent: action.cpu().numpy() for agent, action in actions.items()}
            next_state, rewards, _, _ = self.env.step(step_actions)
            next_state = self.convert_state_to_tensor(next_state)
            # Log
            self.logger.debug(f"Rewards: {rewards}")

            # Append to the batch
            for agent in agents:
                states_dict[agent].append(state[agent])
                actions_dict[agent].append(actions[agent])
                rewards_dict[agent].append(torch.FloatTensor([rewards[agent]]).unsqueeze(0).to(self.device))
                predictions_dict[agent].append(predictions[agent])
                next_states_dict[agent].append(next_state[agent])
                values_dict[agent].append(values[agent])
            
            # # Remove Log - info since it's very fast
            # if iteration % 100 == 0 and iteration:
            #     elapsed = ms_to_time((time.time() - start_timer)*1000)
            #     timer = (((time.time() - start_timer)/(iteration+1))*(self.batch_size - iteration -1))*1000
            #     eta = ms_to_time(timer)
            #
            #     self.logger.info(f"Batch creation: {iteration}/{self.batch_size}, Elapsed: {elapsed}, ETA: {eta} [mm:]ss.ms")
 
            state = copy.deepcopy(next_state)

        self.logger.info(f"Batch of size {self.batch_size} created in {ms_to_time((time.time() - start_timer)*1000)}. [mm:]ss.ms")

        r_temp = []
        for agent, values in rewards_dict.items():
            r_temp.append(round((np.count_nonzero(values)/len(values))*100,2))
        self.logger.info(f"Rewards neq zero: '0' {r_temp[0]}%, '1' {r_temp[1]}%, '2' {r_temp[2]}%, '3' {r_temp[3]}%")
        del r_temp, actions, predictions, rewards, next_state, state

        return states_dict, actions_dict, rewards_dict, predictions_dict, next_states_dict, values_dict

    def train(self, states: dict, actions: dict, rewards: dict, predictions: dict, next_states: dict, values: dict) -> dict:
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
        losses = {'0': [], '1': [], '2': [], '3': []}

        self.logger.warning("For now removing the 'p' agent from the training. In future THIS MUST BE FIXED.")
        self.logger.info(f"Training on {self.batch_size} steps.")
        
        start_timer = time.time()
        for agent in states.keys():
            if agent != 'p':
                # Initialize lists for storing the values and next values
                _values = values[agent][:-1]
                _next_values = values[agent][1:]
                # Log
                self.logger.debug(f"Values: {(len(_values), _values[0].shape)}")
                self.logger.debug(f"Next values: {(len(_next_values), _next_values[0].shape)}")

                # Calculate GAEs and target values
                gaes, target_values = self._get_gaes(rewards[agent], _values, _next_values)
                # Log
                self.logger.debug(f"GAEs: {[round(float(v), 3) for v in gaes]}")
                self.logger.debug(f"Target values: {[round(float(v), 3) for v in target_values]}")

                # Fit the networks
                loss = self.actor.fit(states=states[agent], epochs=self.epochs, batch_size=self.batch_size, predictions=predictions[agent], actions=actions[agent], gaes=gaes)
                losses[agent] = loss
            else:
                self.logger.warning("For now removing the 'p' agent from the training. In future THIS MUST BE FIXED.")
            
        # Should make checkpoint here
        self.logger.info(f"Training took {round(time.time() - start_timer, 2)} seconds.")
        for key, value in losses.items():
            self.logger.info(f"Loss for agent {key}: {round(value[-1], 3)} with min: {round(min(value),3)}, max: {round(max(value),3)} and mean: {round(np.mean(value),3)}")
        self.checkpoint()

        return losses

    def checkpoint(self):
        """
        Save the weights of the Actor and Critic networks.
        """            
        # self.actor.save_weights(os.path.join(self.checkpoint_path, "actor.h5"))
        # self.critic.save_weights(os.path.join(self.checkpoint_path, "critic.h5"))
        self.logger.info("Checkpoint saved.")

    def test(self, episodes: int = 1) -> None:
        """
        Test the agent.
        """
        self.logger.debug(f"Testing the agent for {episodes} episodes...")
        avg_reward = {'0': [], '1': [], '2': [], '3': []}
        for episode in range(episodes):
            state = self.env.reset()
            done = {'__all__': False}
            step = 0
            while not done.get('__all__') or step < self.batch_size:
                actions, _, _ = self.get_actions(state)
                state, rewards, done, _ = self.env.step(actions)
                for agent in ['0', '1', '2', '3']:
                    avg_reward[agent].append(rewards[agent])
                step += 1
            
            self.logger.debug(f"Episode {episode+1}/{episodes} completed.")
            if any([val > 0 for val in rewards.values()]):
                self.logger.debug(f"Average reward: '0' {rewards['0']/step}, '1' {rewards['1']/step}, '2' {rewards['2']/step}, '3' {rewards['3']/step}")
                
        self.logger.debug(f"Testing completed.")

        return avg_reward

