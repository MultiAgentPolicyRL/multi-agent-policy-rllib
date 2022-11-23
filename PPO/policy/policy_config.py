"""
Set up and data check of policy's config
"""
from model.model_config import ModelConfig


class PolicyConfig():
    """
    a
    """
    def __init__(self, action_space, observation_space = None):
        # self.batch_size = batch_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.model_config = ModelConfig(action_space=action_space, observation_space=observation_space)
        self.batch_size = 0

    def set_batch_size(self, batch_size:int, ):
        """
        Sets batch_size dimension
        """
        self.batch_size = batch_size

