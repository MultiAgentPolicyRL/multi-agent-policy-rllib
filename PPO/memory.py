"""
AI-Economist Batch memory. It's used by `algorithm`.
"""

# pylint: disable=no-name-in-module
# pylint: disable=import-error

from dataclasses import dataclass
from utils.timeit import timeit


@dataclass
class BatchMemory:
    def __init__(
        self,
        policy_mapping_function,
        policy_config: dict,
        available_agent_id: list,
    ):
        self.policy_config = policy_config
        self.policy_mapping_function = policy_mapping_function
        # ['0','1','2','3','p']
        self.available_agent_ids = available_agent_id

        self.observation = {key: [] for key in self.available_agent_ids}
        self.policy_action = {key: [] for key in self.available_agent_ids}
        self.policy_probability = {key: [] for key in self.available_agent_ids}
        self.vf_action = {key: [] for key in self.available_agent_ids}
        self.reward = {key: [] for key in self.available_agent_ids}

        """
        Internal structure is like this:
        self.observation = {
            '0': {...},
            '1': {...},
            ...
            'p': {...},
        }
        """

    @timeit
    def reset_memory(self):
        """
        Clears the memory keeping internal baseic data structure.

        Basically it remains like this:
        self.observation = {'0':[], '1':[], '2':[], '3':[], 'p':[]}
        """
        for key in self.available_agent_ids:
            self.observation[key].clear()
            self.policy_action[key].clear()
            self.policy_probability[key].clear()
            self.vf_action[key].clear()
            self.reward[key].clear()

    # @timeit
    def update_memory(
        self,
        observation: dict,
        policy_action: dict,
        policy_probability: dict,
        vf_action: dict,
        reward: dict
    ):
        """
        Splits each input for each agent and appends its data to the correct structure

        Args:
            observation: environment observation with all agents ('0', '1', '2', '3', 'p')
            policy_action: policy's taken action, single for '0', '1', '2', '3', multiple for 'p'
            policy_probability: policy's action distribution for each agent
            vf_action: value function action prediction for each agent
            reward: environment given reward for each agent

        Return:
            nothing
        """
        for key in self.available_agent_ids:
            self.observation[key].append(observation[key])
            self.policy_action[key].append(policy_action[key])
            self.policy_probability[key].append(policy_probability[key])
            self.vf_action[key].append(vf_action[key])
            self.reward[key].append(reward[key])

    @timeit
    def get_memory(self, mapped_key):
        """
        Each memorized input is retrived from the memory and merged by agent

        Return example:
            observations = [observation['0'], observation['1'], observation['2'], observation['3']]
            policy_actions = [policy_action['0'], policy_action['1'], policy_action['2'], ...

        Args:
            key: selected group

        Returns:
            observations, policy_actions, policy_probabilitiess, value_functions,
            rewards, epochs, steps_per_epoch
        """

        observations = []
        policy_actions = []
        policy_probabilitiess = []
        value_functions = []
        rewards = []

        for key in self.available_agent_ids:
            if self.policy_mapping_function(key) == mapped_key:
                observations.extend(self.observation[key])
                policy_actions.extend(self.policy_action[key])
                policy_probabilitiess.extend(self.policy_probability[key])
                value_functions.extend(self.vf_action[key])
                rewards.extend(self.reward[key])

        epochs = self.policy_config[mapped_key].agents_per_possible_policy
        steps_per_epoch = len(observations)/epochs

        return (observations, policy_actions, policy_probabilitiess,
            value_functions, rewards, epochs, steps_per_epoch)
