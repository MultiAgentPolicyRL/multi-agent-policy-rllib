"""
PPO's top level algorithm.
Manages batching and multi-agent training.
"""
# pylint: disable=no-member
# pylint: disable=import-error
# pylint: disable=no-name-in-module
# pylint: disable = consider-using-dict-items
import copy
import logging
import sys

import torch
from utils.timeait import timeit

from algorithm.algorithm_config import AlgorithmConfig
from memory import BatchMemory
from policy.ppo_policy import PPOAgent

general_logger = logging.getLogger("general")
data_logger = logging.getLogger("total_rew")


class PpoAlgorithm(object):
    """
    PPO's top level algorithm.
    Manages batching and multi-agent training.
    """

    def __init__(self, algorithm_config: AlgorithmConfig):
        """
        policy_config: dict,
        available_agent_groups,
        policy_mapping_fun=None

        policy config can be like:

        policy_config: {
            'a' : config or None,
            'p' : config or None,
        }
        """
        self.algorithm_config = algorithm_config

        ###
        # Build dictionary for each policy (key) in `policy_config`
        ###
        self.training_policies = {}
        for key in self.algorithm_config.policies_configs:
            # config injection

            self.training_policies[key]: PPOAgent = PPOAgent(
                self.algorithm_config.policies_configs[key]
            )

        # Setup batch memory
        # FIXME: doesn't work with `self.algorithm_config.policy_mapping_fun` reference
        self.memory = BatchMemory(
            self.algorithm_config.policy_mapping_function,
            self.algorithm_config.policies_configs,
            self.algorithm_config.agents_name,
        )

    def train_one_step(
        self,
        env,
    ):
        """
        Train all Policys.
        PPO's data batch is generated and splitted to each policy following
        `self.policy_mapping_fun` rules. The environment is copied and resetted to avoid
        problems with non-differentiable operations.

        Args:
            env: environment where training is done

        Returns:
            nothing
        """
        # Resetting memory
        self.memory.reset_memory()
        env = env
        # env = copy.deepcopy(env)

        # Collecting data for batching
        self.batch(env)
        # Pass batch to the correct policy to perform training
        for key in self.training_policies:
            self.training_policies[key].learn(*self.memory.get_memory(key))

    # @timeit
    def batch(self, env):
        """
        Generates and memorizes a batch of `self.algorithm_config.batch_size` size.
        Data is stored in `self.memory`

        Args:
            env: environment where batching is done

        Returns:
            nothing
        """
        observation = env.reset()
        steps = 0
        total_actors_reward = 0

        while steps < self.algorithm_config.batch_size:
            # if steps % 100 == 0:
            #     general_logger.debug(f"    step: {steps}")

            # Preprocess observation so that it's made of torch.tensors
            observation = self.data_preprocess(observation=observation)

            # Actor picks an action
            # Returned data are all torch.tensors
            policy_actions, policy_probabilities = self.get_actions(observation)

            # Retrieve new state, rew
            next_observation, reward, _, _ = env.step(policy_actions)

            total_actors_reward += reward["0"] + reward["1"] + reward["2"] + reward["3"]
            # FIXME (?): reward is still a np.array

            # Memorize (state, action, reward) for trainig
            self.memory.update_memory(
                observation=observation,
                policy_action=policy_actions,
                policy_probability=policy_probabilities,
                reward=reward,
            )

            observation = next_observation
            steps += 1

        general_logger.debug("TOTAL REWARD: %.4f", total_actors_reward)
        data_logger.info(f"{total_actors_reward}")

    # @timeit
    def get_actions(self, observation: dict) -> dict:
        """
        Build action dictionary from env observations. Output has thi structure:

                actions: {
                    '0': [...],
                    '1': [...],
                    '2': [...],
                    ...
                    'p': [...]
                }

        FIXME: Planner

        Arguments:
            observation: observation dictionary of the environment, it contains all observations for each agent

        Returns:
            policy_actions dict: predicted actions for each agent
            policy_probability dict: action probabilities for each agent
            vf_actions dict: value function action predicted for each agent
        """

        # Define built memories
        policy_actions, policy_probabilities = {}, {}

        # Bad implementation that works only with agents.
        # To work also with the planner it needs to know which agents are trained so that it can default
        # to something if they are not under a policy. (so 2 different default, one for 'a', one for 'p')
        for key in observation.keys():
            if key != "p":
                (
                    policy_actions[key],
                    policy_probabilities[key],
                ) = self.training_policies[
                    self.algorithm_config.policy_mapping_function(key)
                ].act(
                    observation[key]
                )
            else:
                # tmp to also feed the planner
                policy_actions[key], policy_probabilities[key] = (
                    [
                        torch.zeros((1,)),
                        torch.zeros((1,)),
                        torch.zeros((1,)),
                        torch.zeros((1,)),
                        torch.zeros((1,)),
                        torch.zeros((1,)),
                        torch.zeros((1,)),
                    ],
                    torch.zeros((1,)),
                )

        return policy_actions, policy_probabilities

    def data_preprocess(self, observation: dict) -> dict:
        """
        Takes as an input a dict of np.arrays and trasforms them to Torch.tensors.

        Args:
            observation: observation of the environment

        Returns:
            observation_tensored: same structure of `observation`, but np.arrays are not
                torch.tensors

            observation_tensored: {
                '0': {
                    'var': Tensor
                    ...
                },
                ...
            }
        """
        # observation_tensored = observation

        observation_tensored = {}

        for key in observation:
            # Agents: '0', '1', '2', '3', 'p'
            observation_tensored[key] = {}
            for data_key in observation[key]:
                # Accessing to specific data like 'world-map', 'flat', 'time', ...
                observation_tensored[key][data_key] = (
                    torch.Tensor(observation[key][data_key]).unsqueeze(0).long()
                )
        return observation_tensored
