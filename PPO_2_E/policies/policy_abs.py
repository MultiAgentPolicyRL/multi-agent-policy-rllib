"""
Abstract policy class
"""

from abc import ABC, abstractmethod
from utils.rollout_buffer import RolloutBuffer


class Policy(ABC):
    """
    Abstract policy class
    """

    def __init__(
        self,
        observation_space,
        action_space,
        batch_size,
    ):
        super().__init__()
        # Initialization
        # Environment and PPO parameters
        self.observation_space = observation_space
        self.action_space = action_space  # self.env.action_space.n
        self.batch_size = batch_size  # training epochs

    @abstractmethod
    def act(self, observation: dict):
        """
        Return policy actions
        """
        NotImplementedError("This method must be implemented")

    @abstractmethod
    def learn(
        self,
        rollout_buffer: RolloutBuffer,
        epochs: int,
        steps_per_epoch: int,
    ):
        """
        Update the policy
        """
        NotImplementedError("This method must be implemented")
