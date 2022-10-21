from env_wrapper import RLlibEnvWrapper
from configs.common_config import common_params
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from policy_configs import get_policy_configs
from tf_models import KerasConvLSTM 
import ray

policies, policies_to_train, policy_mapping_fun, logger_creator, trainer_config = get_policy_configs(1)

common_param = common_params()
config = {
    "env_config_dict": common_param["env"]
}

env = RLlibEnvWrapper(config)
register_env("ai-economist-external", lambda _: ExternalMultiAgentEnv(action_space=env.global_action_space, observation_space=env.global_observation_space))
ray.init() 
trainer = PPOTrainer(env="ai-economist-external", config=dict(trainer_config, **{
    "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fun,
                "policies_to_train": ["agent_policy"],
            },
}))

# ppoAgent : PPOTrainer = PPOTrainer(
#         env="ai-economist",
#         config={
#             "env_config": env_config,
#             "seed": final_seed,
#             "multiagent": {
#                 "policies": policies,
#                 "policy_mapping_fn": policy_mapping_fun,
#                 "policies_to_train": ["agent_policy"],
#             },
#             "metrics_smoothing_episodes": trainer_config.get("num_workers")
#             * trainer_config.get("num_envs_per_worker"),
#         },
#         logger_creator=logger_creator
#     )


while True:
    print(trainer.train())