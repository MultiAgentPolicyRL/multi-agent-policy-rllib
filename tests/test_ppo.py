# python -m tests.test_ppo
import tensorflow as tf
import unittest
import numpy as np
from PPO.main_test import get_environment, dict_to_tensor_dict
from PPO.my_own_ppo import ActorModel, CriticModel, PPOAgent

class ModelChecking(unittest.TestCase):
    """
    Testing NN model used in PPO, gets the environemnt from ..PPO.main_test import get_environment
    """
    def setUp(self):
        """
        Test setup
        """
        self.ppo_agent = PPOAgent("test_ppo")
        self.actor_model = ActorModel()
        self.critic_model = CriticModel()
        self.env = get_environment()
        self.obs = self.env.reset()

    def test_feeding_actor_nn(self):
        """
        Feeds actorModel network
        """
        actor_value = self.actor_model.predict(self.obs['0'])
        self.assertIsNot(actor_value, None)

    # def test_feeding_critic_nn(self):
    #     """
    #     Feeds criticModel network FIXME
    #     """
    #     critic_value = self.critic_model.predict(self.obs['0'])
    #     self.assertIsNot(critic_value, None)

    def test_build_action_dict(self):
        """
        Check if build_action_dict works
        """
        actions = self.ppo_agent.build_action_dict(self.obs)
        self.assertIsInstance(actions, dict)
        
if __name__ == "__main__":
    unittest.main()
