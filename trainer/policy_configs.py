"""
Implement and return policies, policies_mapping_fn
"""
import logging
import sys
import time

from ray.tune.logger import NoopLogger, pretty_print
from ray.tune.registry import register_env

from configs.common_config import common_params
from configs.phase1 import ppo_policy_params_1
from env_wrapper import RLlibEnvWrapper

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


def get_policy_configs(phase: int = 1):
    """Finalize the trainer config by combining the sub-configs.
    Args:
        `phase`:int:default=1 Training phase

    Returns:
        `policies`, `policies_to_train`, `policy_mapping_fun`, `logger_creator`"""
    common_param = common_params()
    trainer_config = common_param["trainer"]

    # Env config
    env_config = {
        "env_config_dict": common_param["env"],
        "num_envs_per_worker": trainer_config["num_envs_per_worker"]
    }

    # Seed
    if trainer_config["seed"] is None:
        start_seed = int(time.time())
    else:
        start_seed = int(trainer_config["seed"])

    final_seed = int((start_seed) % (2 ** 16)) * 1000
    logger.info("seed (final): %s", final_seed)

    # Multiagent Policies
    dummy_env = RLlibEnvWrapper(env_config)
    register_env("ai-economist", lambda _: RLlibEnvWrapper(env_config))


    # Policy tuples for agent/planner policy types
    if (phase == 1):
        ppoPolicyParams = ppo_policy_params_1()
    elif (phase == 2):
        NotImplementedError("Phase 2 not implemented yet")
    else:
        ValueError("phase must be 1 or 2")

    agent_policy_tuple = (
        None,
        dummy_env.observation_space,
        dummy_env.action_space,
        ppoPolicyParams["ppo_agent_policy"],
    )

    planner_policy_tuple = (
        None,
        dummy_env.observation_space_pl,
        dummy_env.action_space_pl,
        ppoPolicyParams["ppo_planner_policy"],
    )

    policies = {"a": agent_policy_tuple, "p": planner_policy_tuple}

    def policy_mapping_fun(i):
        if str(i).isdigit() or i == "a":
            return "a"
        return "p"

    # Which policies to train
    if common_param["general"]["train_planner"]:
        policies_to_train = ["a", "p"]
    else:
        policies_to_train = ["a"]

    def logger_creator(config):
        return NoopLogger({}, "/tmp")

    return policies, policies_to_train, policy_mapping_fun, logger_creator
