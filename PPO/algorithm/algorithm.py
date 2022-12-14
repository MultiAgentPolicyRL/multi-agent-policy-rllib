"""
PPO's top level algorithm.
Manages batching and multi-agent training.
"""
# pylint: disable=no-member
# pylint: disable=import-error
# pylint: disable=no-name-in-module

import copy
import sys

import torch
from utils.timeit import timeit

from algorithm.algorithm_config import AlgorithmConfig
from memory import BatchMemory
from policy.policy import PPOAgent


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
            self.algorithm_config.env,
        )

    def train_one_step(
        self,
        env,
    ):
        """
        Train all Policys
        Here PPO's Minibatch is generated and splitted to each policy, following
        `policy_mapping_fun` rules
        """
        # Resetting memory
        self.memory.reset_memory()
        env = copy.deepcopy(env)

        # Collecting data for batching
        self.batch(env)
        # Pass batch to the correct policy to perform training
        for key in self.training_policies:  # pylint: disable = consider-using-dict-items
            # logging.debug(f"Training policy {key}")
            self.training_policies[key].learn(*self.memory.get_memory(key))

        

    @timeit
    def batch(self, env):
        observation = env.reset()
        steps = 0

        # FIXME: add correct data type
        vf_prediction_old = {
            "0": 0,
            "1": 0,
            "2": 0,
            "3": 0,
            "p": 0,
        }

        while steps < self.algorithm_config.batch_size:
            # if steps % 100 == 0:
            #     logging.debug(f"    step: {steps}")

            # Actor picks an action
            (
                policy_action,
                policy_action_onehot,
                policy_prediction,
                vf_prediction,
            ) = self.get_actions(observation)

            # Retrieve new state, rew
            next_observation, reward, _, _ = env.step(policy_action)

            # Memorize (state, action, reward) for trainig
            self.memory.update_memory(
                observation=observation,
                next_observation=next_observation,
                policy_action_onehot=policy_action_onehot,
                reward=reward,
                policy_prediction=policy_prediction,
                vf_prediction=vf_prediction,
                vf_prediction_old=vf_prediction_old,
            )
            # sys.exit()

            observation = next_observation
            vf_prediction_old = vf_prediction
            steps += 1

    # @timeit
    def get_actions(self, obs: dict) -> dict:
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
            obs: observation dictionary of the environment, it contains all observations for each agent

        Returns:
            actions dict: actions for each agent
        """

        actions, actions_onehot, predictions, values = {}, {}, {}, {}
        for key in obs.keys():
            if key != "p":
                # print(self._policy_mapping_function(key))
                (
                    actions[key],
                    actions_onehot[key],
                    predictions[key],
                    values[key],
                ) = self.training_policies[
                    self.algorithm_config.policy_mapping_function(key)
                ].act(
                    obs[key]
                )
            else:
                # tmp to also feed the planner
                actions[key], actions_onehot[key], predictions[key], values[key] = (
                    [torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,)), torch.zeros((1,))],
                    torch.zeros((1,)),
                    torch.zeros((1,)),
                    torch.zeros((1,)),
                )
        # logging.debug(actions)
        return actions, actions_onehot, predictions, values
