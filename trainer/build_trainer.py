from env_wrapper import RLlibEnvWrapper
from tf_models import KerasConvLSTM
from configs.common_config import common_params
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.tune.registry import register_env
from policy_configs import get_policy_configs
from ray.rllib.agents.ppo import PPOTrainer
import time

def build_trainer():
    common_param = common_params()

    policies, policies_to_train, policy_mapping_fn, logger_creator, dummy_env = get_policy_configs(1)

    seed = int(time.time())

    trainer_config = common_param["trainer"]

    env_config = {
            "env_config_dict": common_param["env"],
            "num_envs_per_worker": trainer_config["num_envs_per_worker"]
        }

    ppoAgent : PPOTrainer = PPOTrainer(
        env="ai-economist",
        config={
            "env_config": env_config,
            "seed": seed,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["agent_policy"],
            },
        },
    )

    return ppoAgent
        