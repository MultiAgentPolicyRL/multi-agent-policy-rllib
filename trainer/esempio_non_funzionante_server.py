from configs.common_config import common_params
import os

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.logger import pretty_print
from tf_models import KerasConvLSTM

from policy_configs import get_policy_configs

SERVER_ADDRESS = "localhost"
SERVER_PORT = 9900

if __name__ == "__main__":
    ray.init()
    policies, policies_to_train, policy_mapping_fn, logger_creator = get_policy_configs(
        1)

    connector_config = {
        # Use the connector server to generate experiences.
        "input": (
            lambda ioctx: PolicyServerInput( \
                ioctx, SERVER_ADDRESS, SERVER_PORT)
        ),
    }

    def _input(ioctx):
        # We are remote worker or we are local worker with num_workers=0:
        # Create a PolicyServerInput.
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                SERVER_ADDRESS,
                SERVER_PORT + ioctx.worker_index -
                (1 if ioctx.worker_index > 0 else 0),
            )
        # No InputReader (PolicyServerInput) needed.
        else:
            return None

    config = common_params()["trainer"]
    # config.update(
    #     {
    #         "env": "ai-economist",
    #         "input": _input,
    #         "num_workers": 0,
    #         # Disable OPE, since the rollouts are coming from online clients.
    #         "input_evaluation": [],
    #         # Multi-agent setup for the given env.
    #         "multiagent": {
    #             "policies": policies,
    #             "policies_to_train": policies_to_train,
    #             "policy_mapping_fn": policy_mapping_fn,
    #         },
    #     }
    # )

#    algo = PPOTrainer(config=config)
    # algo = PPOTrainer(
    #     env="ai-economist",
    #     config=dict(
    #         connector_config, **{
    #                 "sample_batch_size": 1000,
    #                 "train_batch_size": 4000,
    #             }))
    algo = PPOTrainer(env="ai-economist", config=dict({"sample_batch_size": 1000,
                                                       "train_batch_size": 4000,
                                                       "input": _input,
                                                       "num_workers": 2,
                                                       # Disable OPE, since the rollouts are coming from online clients.
                                                       "input_evaluation": [],
                                                       # Multi-agent setup for the given env.
                                                       "multiagent": {
                                                           "policies": policies,
                                                           "policies_to_train": policies_to_train,
                                                           "policy_mapping_fn": policy_mapping_fn,
                                                       }, }))
