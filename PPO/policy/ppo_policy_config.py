"""
Set up and data check of policy's config
"""
from model.model_config import ModelConfig


class PpoPolicyConfig:
    """
    a
    """

    def __init__(self, action_space, observation_space=None, name: str = ""):
        # self.batch_size = batch_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.model_config = ModelConfig(
            action_space=action_space, observation_space=observation_space, name=name
        )
        self.batch_size = 0
        self.agents_per_possible_policy = 0
        self.num_workers = 1

    def set_batch_size_and_agents_per_possible_policy(
        self, batch_size: int, agents_per_possible_policy, num_workers: int = 1
    ):
        """
        Sets batch_size dimension
        """
        self.batch_size = batch_size
        self.agents_per_possible_policy = agents_per_possible_policy
        self.num_workers = num_workers
