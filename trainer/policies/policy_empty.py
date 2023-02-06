"""
Empty policy to manage an `unmanaged` policy. Used when an actor (or set of actors)
is not under a real policy. Every action is set to 0.
"""
# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=import-error

from typing import Tuple
from trainer.policies import Policy
from trainer.utils import RolloutBuffer, exec_time
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
        # TODO: revert
        # for _ in self.action_space:
        #   actions.append(torch.zeros((1,)))

        for _ in self.action_space:
            actions.append(0)

        if len(actions) == 1:
            actions = actions[0]

        return actions, 0

    def learn(self, rollout_buffer: RolloutBuffer) -> Tuple[float, float]:
        """
        This policy doesn't have to learn anything. It will just do nothing.
        """
        return 0.0, 0.0

    def get_weights(self):
        """
        Get policy weights.

        Return:
            weights
        """
        return {"a": None, "c": None}

    def set_weights(self, weights) -> None:
        """
        Set policy weights.

        Return:
            weights
        """
