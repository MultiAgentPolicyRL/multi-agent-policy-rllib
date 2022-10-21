from env_wrapper import RLlibEnvWrapper
from configs.common_config import common_params
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.tune.registry import register_env

common_param = common_params()
config = {
    "env_config_dict": common_param["env"]
}

env = RLlibEnvWrapper(config)
# print(env.global_action_space)


register_env("ai-economist-external", lambda _: ExternalMultiAgentEnv(action_space=env.global_action_space, observation_space=env.global_observation_space))

