"""
    Buffer used to store batched data
"""
# All of this can be improved preallocating all the memory and stuff like that.
from typing import List

import torch
from tensordict import TensorDict

from trainer.utils import exec_time


class RolloutBuffer:
    """
    Buffer used to store batched data
    """

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
        """
        Extend this buffer with `other`'s data
        """
        self.actions.extend(other.actions)
        self.logprobs.extend(other.logprobs)
        self.states.extend(other.states)
        self.rewards.extend(other.rewards)
        self.is_terminals.extend(other.is_terminals)

    # @exec_time
    def to_tensor(self) -> dict:
        """
        Transforms all data of this buffer to `torch.Tensor`
        """
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
