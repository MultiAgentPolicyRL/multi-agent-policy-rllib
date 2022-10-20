import unittest
from trainer.env_wrapper import RLlibEnvWrapper
from trainer.configs.common_config import env_params


class TestEnvBuild(unittest.TestCase):
    def testBuild(self):
        config = {
            "env_config_dict": env_params()
        }
        env = RLlibEnvWrapper(config)


if __name__ == '__main__':
    unittest.main()