"""
    Buffer used to store batched data
"""
# FIXME: all of this can be improved preallocating all the
# memory and stuff like that.
from typing import List
from utils import exec_time
import torch
class RolloutBuffer:
    """
    Buffer used to store batched data
    """

    def __init__(self, batch_size, n_agents):
        self.actions = None
        self.states = None
        self.logprobs = None
        self.rewards = None
        self.is_terminals = None
        self.batch_size = batch_size
        self.n_agents = n_agents

    def clear(self):
        """
        Clear the buffer
        """
        self.actions = None
        self.states = None
        self.logprobs = None
        self.rewards = None
        self.is_terminals = None


class Memory:
    """
    Batch memory used during batching and training
    FIXME: we can simplify this memory just by creating `n_agents` `RolloutBuffers`.
    FIXME: when 'get' is called a list of RolloutBuffers is returned (simpler than that is impossible.)
    """

    def __init__(
        self,
        policy_mapping_fun,
        available_agent_id: list,
        policy_size: dict,
        batch_size: int,
        device: str,
    ):
        self.device = device
        self.policy_mapping_fun = policy_mapping_fun
        # 0,1,2,3,p
        self.available_agent_ids = available_agent_id
        # a: 4, p: 1
        self.policy_size = policy_size

        self.action = {key: [] for key in self.available_agent_ids}
        self.logprob = {key: [] for key in self.available_agent_ids}
        self.state = {key: [] for key in self.available_agent_ids}
        self.reward = {key: [] for key in self.available_agent_ids}
        self.is_terminal = {key: [] for key in self.available_agent_ids}

        # self.rollout_buffer = Dict[str, RolloutBuffer]
        self.rollout_buffer = {}
        for key in self.available_agent_ids:
            self.rollout_buffer[key] = RolloutBuffer(
                batch_size=batch_size, n_agents=policy_size[self.policy_mapping_fun(key)]
            )

    def clear(self):
        """
        Clears the memory while keeping the internal basic data structure.
        It remains like this:
        self.actions = {'0':[], '1':[], '2':[], '3':[], 'p':[]}
        """

        for key in self.available_agent_ids:
            self.action[key].clear()
            self.logprob[key].clear()
            self.state[key].clear()
            self.reward[key].clear()
            self.is_terminal[key].clear()

    def update(
        self, action: dict, logprob: dict, state: dict, reward: dict, is_terminal: dict
    ):
        """
        Splits each input for each agent in `self.available_agent_ids` and appends
        its data to the correct structure.
        Args:
            action: taken action
            logprob: log probability of the taken action
            state: environment observation
            reward: agent's reward for this action
            is_terminal: if this is the last action for this environment
        """
        if is_terminal['__all__'] == False:
            is_terminal=False
        else:
            is_terminal=True

        for key in self.available_agent_ids:
            self.action[key].append(action[key])
            self.logprob[key].append(logprob[key])
            self.state[key].append(state[key])
            self.reward[key].append(reward[key])
            self.is_terminal[key].append(is_terminal)

    @exec_time
    def get(self, mapped_key) -> List[RolloutBuffer]:
        """
        Each memorized input is retrived from the memory and merged by agent.
        Return example:
            observations = [observation['0'], observation['1'], observation['2'], observation['3']]
            policy_actions = [policy_action['0'], policy_action['1'], policy_action['2'], ...
        Args:
            mapped_key: selected group
        Returns:
            RolloutBuffer filled with the selected batch
        """

        rollout_buffers = []
        for key in self.available_agent_ids:

            if self.policy_mapping_fun(key) == mapped_key:
                self.rollout_buffer[key].clear()
                self.rollout_buffer[key].actions = torch.Tensor(self.action[key]).to(self.device)
                self.rollout_buffer[key].logprobs = torch.Tensor(self.logprob[key]).to(self.device)
                self.rollout_buffer[key].states = torch.stack(self.state[key]).to(self.device)
                self.rollout_buffer[key].rewards = torch.Tensor(self.reward[key]).to(self.device)
                self.rollout_buffer[key].is_terminals = torch.Tensor(self.is_terminal[key]).to(self.device)

                rollout_buffers.append(self.rollout_buffer[key])

        return rollout_buffers