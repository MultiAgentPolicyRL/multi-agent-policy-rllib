import sys
sys.path.append('..')

import tensorflow as tf
import unittest
from PPO.my_own_ppo import ActorModel, CriticModel
from PPO.main_test import get_environment, dict_to_tensor_dict



class ModelChecking(unittest.TestCase):
    """
    Testing NN model used in PPO, gets the environemnt from ..PPO.main_test import get_environment
    """

    def test_feeding_neural_network(self):
        """
        Feeds both actorModel and criticModel networks
        """
        actor_model = ActorModel()
        critic_model = CriticModel()
        env = get_environment()
        obs = env.reset()
        obs = dict_to_tensor_dict(obs['0'])
        actor_value = actor_model.predict(obs)
        critic_value = critic_model.predict(obs)
        self.assertIsNot(actor_value, None)
        self.assertIsNot(critic_value, None)


if __name__ == "__main__":
    unittest.main()
