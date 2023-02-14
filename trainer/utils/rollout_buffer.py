"""
    Buffer used to store batched data
"""
# FIXME: all of this can be improved preallocating all the
# memory and stuff like that.
from typing import List

import numpy as np
import torch
from tensordict import TensorDict

from trainer.utils import exec_time


class RolloutBuffer:
    """
    Buffer used to store batched data
    """
    # TODO: move to_tensor inside each policy.

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        """
        Clear the buffer
        """
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

    def update(self, action, logprob, state, reward, is_terminal):
        """
        Append single observation to this buffer.

        Args:
            action
            logprob
            state
            reward
            is_terminal
        """
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.states.append(state)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def extend(self, other):
        self.actions.extend(other.actions)
        self.logprobs.extend(other.logprobs)
        self.states.extend(other.states)
        self.rewards.extend(other.rewards)
        self.is_terminals.extend(other.is_terminals)

    # @exec_time
    def to_tensor(self):
        buffer = RolloutBuffer()
        buffer.actions = torch.tensor((self.actions))
        buffer.logprobs = torch.tensor((self.logprobs))
        buffer.rewards = torch.tensor((self.rewards))
        buffer.is_terminals = torch.tensor((self.is_terminals))

        states = []
        for state in self.states:
            states.append(TensorDict(state, batch_size=[]))
            # states.append(self._dict_to_tensor_dict(state))

        buffer.states = torch.stack(states)
        return buffer

    def _dict_to_tensor_dict(self, data: dict):
        observation_tensored = {}

        for key in data.keys():
            observation_tensored[key] = torch.from_numpy(data[key])

        return observation_tensored


# class Memory:
#     """
#     Batch memory used during batching and training
#     FIXME: we can simplify this memory just by creating `n_agents` `RolloutBuffers`.
#     FIXME: when 'get' is called a list of RolloutBuffers is returned (simpler than that is impossible.)
#     """

#     def __init__(
#         self,
#         policy_mapping_fun,
#         available_agent_id: list,
#         policy_size: dict,
#         batch_size: int,
#         device: str,
#     ):
#         self.device = device
#         self.policy_mapping_fun = policy_mapping_fun
#         # 0,1,2,3,p
#         self.available_agent_ids = available_agent_id
#         # a: 4, p: 1
#         self.policy_size = policy_size

#         self.action = {key: [] for key in self.available_agent_ids}
#         self.logprob = {key: [] for key in self.available_agent_ids}
#         self.state = {key: [] for key in self.available_agent_ids}
#         self.reward = {key: [] for key in self.available_agent_ids}
#         self.is_terminal = {key: [] for key in self.available_agent_ids}

#         # self.rollout_buffer = Dict[str, RolloutBuffer]
#         # self.rollout_buffer = {}
#         # for key in self.available_agent_ids:
#         #     self.rollout_buffer[key] = RolloutBuffer(
#         #         batch_size=batch_size,
#         #         n_agents=policy_size[self.policy_mapping_fun(key)],
#         #     )

#     def append(self, other):
#         """
#         Temporary way (I'd prefer a 'sum' method) to append another filled memory to this one.
#         """
#         for key in self.available_agent_ids:
#             self.action[key].extend(other.action[key])
#             self.logprob[key].extend(other.logprob[key])
#             self.state[key].extend(other.state[key])
#             self.reward[key].extend(other.reward[key])
#             self.is_terminal[key].extend(other.is_terminal[key])

#     def clear(self):
#         """
#         Clears the memory while keeping the internal basic data structure.
#         It remains like this:
#         self.actions = {'0':[], '1':[], '2':[], '3':[], 'p':[]}
#         """
#         for key in self.available_agent_ids:
#             self.action[key].clear()
#             self.logprob[key].clear()
#             self.state[key].clear()
#             self.reward[key].clear()
#             self.is_terminal[key].clear()

#     def update(
#         self, action: dict, logprob: dict, state: dict, reward: dict, is_terminal: dict
#     ):
#         """
#         Splits each input for each agent in `self.available_agent_ids` and appends
#         its data to the correct structure.
#         Args:
#             action: taken action
#             logprob: log probability of the taken action
#             state: environment observation
#             reward: agent's reward for this action
#             is_terminal: if this is the last action for this environment
#         """
#         if is_terminal["__all__"] == False:
#             is_terminal = False
#         else:
#             is_terminal = True

#         for key in self.available_agent_ids:
#             self.action[key].append(action[key])
#             self.logprob[key].append(logprob[key])
#             self.state[key].append(state[key])
#             self.reward[key].append(reward[key])
#             self.is_terminal[key].append(is_terminal)

#     @exec_time
#     def get(self, mapped_key) -> List[RolloutBuffer]:
#         """
#         Each memorized input is retrived from the memory and merged by agent.
#         Return example:
#             observations = [observation['0'], observation['1'], observation['2'], observation['3']]
#             policy_actions = [policy_action['0'], policy_action['1'], policy_action['2'], ...
#         Args:
#             mapped_key: selected group
#         Returns:
#             RolloutBuffer filled with the selected batch
#         """

#         rollout_buffers = []
#         for key in self.available_agent_ids:

#             if self.policy_mapping_fun(key) == mapped_key:
#                 self.rollout_buffer[key].clear()
#                 self.rollout_buffer[key].actions = torch.Tensor(self.action[key]).to(
#                     self.device
#                 )
#                 self.rollout_buffer[key].logprobs = torch.Tensor(self.logprob[key]).to(
#                     self.device
#                 )
#                 self.rollout_buffer[key].states = torch.stack(self.state[key]).to(
#                     self.device
#                 )
#                 self.rollout_buffer[key].rewards = torch.Tensor(self.reward[key]).to(
#                     self.device
#                 )
#                 self.rollout_buffer[key].is_terminals = torch.Tensor(
#                     self.is_terminal[key]
#                 ).to(self.device)

#                 rollout_buffers.append(self.rollout_buffer[key])

#         return rollout_buffers
