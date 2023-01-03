"""
Empty policy to manage an `unmanaged` policy. Used when an actor (or set of actors)
is not under a real policy. Every action is set to 0.
"""
# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=import-error

from policies.policy_abs import Policy
from utils.rollout_buffer import RolloutBuffer
import torch


class EmptyPolicy(Policy):
    """
    Empty policy to manage an `unmanaged` policy. Used when an actor (or set of actors)
    is not under a real policy. Every action is set to 0.
    """

    def __init__(self, observation_space, action_space):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )

    def act(self, observation: dict):
        """
        Returns this policy actions and a useless placeholder here only to do
        not manage in a smart way all the multi-agent-policy system.
        """
        actions = []

        for _ in self.action_space:
            actions.append(torch.zeros((1,)))

        if len(actions) == 1:
            actions = actions[0]

        return actions, torch.zeros((1,))

    def learn(self, rollout_buffer: RolloutBuffer, epochs: int, steps_per_epoch: int):
        """
        This policy doesn't have to learn anything. It will just do nothing.
        """
