import os
import copy
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Union

###
from ai_economist_ppo_dt.torch import LSTMModel
from ai_economist_ppo_dt.utils import get_basic_logger, ms_to_time, time_it
from ai_economist.foundation.base.base_env import BaseEnvironment


### Choose:
from random import SystemRandom
random = SystemRandom()

#import random

class PPO():
    def __init__(self, env: BaseEnvironment, action_space: int, seed: int = None, batch_size: int = 32, log_level: int = logging.INFO, log_path: str = None) -> None:
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

        if batch_size > 1000:
            self.logger.warning(f"Batch size is very large: {batch_size}. This may cause memory issues (in particular exponential storage time).")

        datetime = log_path.split("/")[-1][:-4]
        self.checkpoint_path = os.path.join(os.getcwd(), "checkpoints", datetime,)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.action_dim = {'a': 1, 'p': env.all_agents[4].action_spaces.shape[0]}
        self.action_space = {'a': env.all_agents[0].action_spaces, 'p': np.sum(env.all_agents[4].action_spaces)}

        ### Initialize Actor and Critic networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = LSTMModel(initial_state['0'], name="AgentLSTM", output_size=self.action_space.get('a'), log_level=20, log_path=log_path, device=self.device).to(self.device)
        self.planner = LSTMModel(initial_state['0'], name="PlannerLSTM", output_size=self.action_space.get('p'), log_level=20, log_path=log_path, device=self.device).to(self.device)
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

    def _get_gaes(self, rewards: Union[list, np.ndarray], values: Union[list, np.ndarray], next_values: Union[list, np.ndarray], gamma:float=0.998, lamda:float=0.98, normalize:bool=True,) -> np.ndarray:
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

        target = torch.FloatTensor(gaes) + values

        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return torch.FloatTensor(gaes).to(self.device), torch.FloatTensor(target).to(self.device)

    def _act(self, state: np.ndarray, agent: str = 'a') -> np.ndarray:
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
            # Get the input state as a tensor for the input layer (world-map and flat features)
            input_state = [
                torch.FloatTensor(state["world-map"]).unsqueeze(0).to(self.device),
                torch.FloatTensor(state["world-idx_map"]).unsqueeze(0).to(self.device),
                torch.FloatTensor(state["time"]).unsqueeze(0).to(self.device),
                torch.FloatTensor(state["flat"]).unsqueeze(0).to(self.device),
                torch.FloatTensor(state["action_mask"]).unsqueeze(0).to(self.device),
            ]
            # Log
            self.logger.debug(f"Input state: {(input_state[0].shape[1:], input_state[1].shape[1:])}")
            # Get the prediction from the Actor network
            with torch.no_grad():
                logits, value = self.actor(input_state)
            prediction = torch.squeeze(logits)
            # Log the prediction
            self.logger.debug(f"Prediction: {[round(float(p), 3) for p in prediction]}")

            # Sample an action from the prediction distribution
            action = torch.FloatTensor(random.choices(np.arange(self.action_space[agent]), weights=prediction.detach().numpy())).to(self.device)
            # Log
            self.logger.debug(f"Action: {action}")

            # One-hot encode the action
            one_hot_action = torch.zeros([self.action_space[agent]]).to(self.device)
            one_hot_action[int(action.item())] = 1
            # Log
            self.logger.debug(f"One-hot action: {np.where(one_hot_action == 1)[0]}")
        else:
            action = [random.randint(0,21) for _ in range(7)]
            one_hot_action = torch.zeros([self.action_space[agent]]).to(self.device)
            one_hot_action[action] = 1
            prediction = torch.zeros([self.action_space[agent]]).to(self.device)
        
        return action, one_hot_action, prediction

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
            # Get the action, one-hot encoded action, and prediction
            action, one_hot_action, prediction = self._act(states[agent], 'a' if agent != 'p' else 'p')
            # Log
            self.logger.debug(f"Agent {agent} action: {action}")
            
            # Store actions, one-hot encoded actions, and predictions
            actions[agent] = action
            actions_one_hot[agent] = one_hot_action
            predictions[agent] = prediction
                
        return actions, actions_one_hot, predictions

    def train(self, states: dict, actions: dict, rewards: dict, predictions: dict, next_states: dict,) -> dict:
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
        losses = {'0': {'actor': [], 'critic': []}, '1': {'actor': [], 'critic': []}, '2': {'actor': [], 'critic': []}, '3': {'actor': [], 'critic': []}}

        self.logger.warning("For now removing the 'p' agent from the training. In future THIS MUST BE FIXED.")
        self.logger.info(f"Training on {self.batch_size} steps.")
        
        start_timer = time.time()
        for agent in states.keys():
            # Inputs for the Critic network predictions
            input_states = []
            input_next_states = []
            for s, ns in zip(states[agent], next_states[agent]):
                input_states.append([torch.FloatTensor(s["world-map"]).unsqueeze(0).to(self.device), torch.FloatTensor(s["flat"]).unsqueeze(0).to(self.device)])
                input_next_states.append([torch.FloatTensor(ns["world-map"]).unsqueeze(0).to(self.device), torch.FloatTensor(ns["flat"]).unsqueeze(0).to(self.device)])
            
            self.logger.debug(f"Input states: {(input_states[0][0].shape[1:], input_states[0][1].shape[1:])}")
            self.logger.debug(f"Input next states: {(input_next_states[0][0].shape[1:], input_next_states[0][1].shape[1:])}")

            # Initialize lists for storing the values and next values
            values = []
            next_values = []
            for i_s, i_ns in zip(input_states, input_next_states):
                values.append(self.critic(i_s))
                next_values.append(self.critic(i_ns))
            
            # Convert lists to numpy arrays and squeeze (reshape to 1D)
            values = torch.FloatTensor([torch.squeeze(v) for v in values]).to(self.device)
            next_values = torch.FloatTensor([torch.squeeze(v) for v in next_values]).to(self.device)
            # Log
            self.logger.debug(f"Values: {[round(float(v), 3) for v in values]}")
            self.logger.debug(f"Next values: {[round(float(v), 3) for v in next_values]}")

            # Calculate GAEs and target values
            gaes, target_values = self._get_gaes(rewards[agent], values, next_values)
            # Log
            self.logger.debug(f"GAEs: {[round(float(v), 3) for v in gaes]}")
            self.logger.debug(f"Target values: {[round(float(v), 3) for v in target_values]}")

            for batch in range(self.batch_size):
                if self.logger.level == logging.DEBUG:
                    torch.autograd.set_detect_anomaly(True)
                    
                # Get y_true for the Actor network
                y_true = [gaes[batch], predictions[agent][batch], torch.FloatTensor(actions[agent][batch])]
                # y_true = torch.FloatTensor(np.hstack([gaes, predictions[agent], actions[agent]]))
                self.logger.debug(f"Actor y_true: {len(y_true)}")

                # Get x values for the Actor&Critic network and train
                world_map = torch.FloatTensor(states[agent][batch]['world-map']).unsqueeze(0).to(self.device)
                flat = torch.FloatTensor(states[agent][batch]['flat']).unsqueeze(0).to(self.device)

                # Log
                self.logger.debug(f"Actor&Critic World map: {world_map.shape}")
                self.logger.debug(f"Actor&Critic Flat: {flat.shape}")

                # Fit the Actor network
                actor_output = self.actor([world_map, flat])
                self.logger.debug(f"Actor output: {actor_output.shape}")

                # Calculate the loss for the Actor network
                actor_loss = self.actor.my_loss(actor_output, y_true)
                self.logger.debug(f"Actor loss: {actor_loss}")

                # Backpropagate the loss
                actor_loss.backward()

                # Update the Actor network
                self.actor.optimizer.step()
                
                
                # Get y_true for the Critic network
                # y_true = values[batch]
                y_true = [target_values[batch], values[batch]]
                #y_true.requires_grad_(True)
                
                # Fit the Critic network
                critic_output = self.critic([world_map, flat])
                self.logger.debug(f"Critic output: {critic_output.shape}")

                # Calculate the loss for the Critic network
                #critic_loss = torch.nn.functional.mse_loss(critic_output, y_true)
                #critic_loss.requires_grad_(True)
                critic_loss = self.critic.my_loss(critic_output, y_true)
                self.logger.debug(f"Critic loss: {critic_loss}")

                # Backpropagate the loss
                critic_loss.backward()

                # Update the Critic network
                self.critic.optimizer.step()

                # Store losses
                losses[agent]['actor'].append(actor_loss.item())
                losses[agent]['critic'].append(critic_loss.item())

                del actor_output, critic_output, actor_loss, critic_loss, y_true, world_map, flat

            del input_states, input_next_states, values, next_values, gaes, target_values
        # Should make checkpoint here
        self.logger.info(f"Training took {round(time.time() - start_timer, 2)} seconds.")
        self.checkpoint()

        return losses

    def checkpoint(self):
        """
        Save the weights of the Actor and Critic networks.
        """            
        # self.actor.save_weights(os.path.join(self.checkpoint_path, "actor.h5"))
        # self.critic.save_weights(os.path.join(self.checkpoint_path, "critic.h5"))
        self.logger.info("Checkpoint saved.")
    
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

        self.logger.info(f"Batch of size {self.batch_size} created in {ms_to_time((time.time() - start_timer)*1000)}. [mm:]ss.ms")

        r_temp = []
        for agent, values in rewards_dict.items():
            r_temp.append(round((np.count_nonzero(values)/len(values))*100,2))
        self.logger.info(f"Rewards neq zero: '0' {r_temp[0]}%, '1' {r_temp[1]}%, '2' {r_temp[2]}%, '3' {r_temp[3]}%")
        del r_temp, actions, predictions, rewards, next_state, state

        return states_dict, actions_dict, rewards_dict, predictions_dict, next_states_dict, None, None

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

