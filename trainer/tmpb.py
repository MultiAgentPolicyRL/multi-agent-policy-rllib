# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import argparse
import copy
import logging
from multiprocessing import dummy
import os
import sys
import time

import ray
import yaml
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.tune.logger import NoopLogger, pretty_print
from ray.tune.registry import register_env


from env_wrapper import RLlibEnvWrapper
from tf_models import KerasConvLSTM  # used by config.yaml
from utils import remote, saving

ray.init(log_to_driver=False)

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


class AieExternalMultiAgentEnv(ExternalMultiAgentEnv):
    def __init__(self, env: RLlibEnvWrapper):
        ExternalMultiAgentEnv.__init__(
            self,
            action_space=env.global_action_space,
            observation_space=env.global_observation_space,
            max_concurrent=100,
        )
        self.env = copy.deepcopy(env)

    def seed(self, seed):
        self.env.seed(seed=seed)

    def run(self):
        eid = self.start_episode()
        obs = self.env.reset()

        counter = 0
        while True:
            action_dict = self.get_action(eid, obs)
            self.log_action(eid, obs, action_dict)
            obs, rew, done, info = self.env.step(action_dict)
            self.log_returns(eid, rew)
            logger.info(f"Step: {counter}, Rew: {rew}")
            counter += 1

            if done["__all__"] == True:
                counter = 0
                logger.info(f"Step: {counter}, done:{done}")
                self.end_episode(eid, obs)
                obs = self.env.reset()
                eid = self.start_episode()


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir", type=str, help="Path to the directory for this run."
    )

    args = parser.parse_args()
    run_directory = args.run_dir

    config_path = os.path.join(args.run_dir, "config.yaml")
    assert os.path.isdir(args.run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    return run_directory, run_configuration


def build_trainer(run_configuration):
    logger.info("Building trainer")
    """Finalize the trainer config by combining the sub-configs."""
    trainer_config = run_configuration.get("trainer")

    # === Env ===
    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": 0,
    }

    # === Seed ===
    if trainer_config["seed"] is None:
        try:
            start_seed = int(run_configuration["metadata"]["launch_time"])
        except KeyError:
            start_seed = int(time.time())
    else:
        start_seed = int(trainer_config["seed"])

    final_seed = int(start_seed % (2 ** 16)) * 1000
    logger.info("seed (final): %s", final_seed)

    # === Multiagent Policies ===
    dummy_env = RLlibEnvWrapper(env_config)

    # Policy tuples for agent/planner policy types
    agent_policy_tuple = (
        None,
        dummy_env.observation_space,
        dummy_env.action_space,
        run_configuration.get("agent_policy"),
    )
    planner_policy_tuple = (
        None,
        dummy_env.observation_space_pl,
        dummy_env.action_space_pl,
        run_configuration.get("planner_policy"),
    )

    policies = {"a": agent_policy_tuple, "p": planner_policy_tuple}

    def policy_mapping_fun(i):
        if str(i).isdigit() or i == "a":
            return "a"
        return "p"

    # Which policies to train
    if run_configuration["general"]["train_planner"]:
        policies_to_train = ["a", "p"]
    else:
        policies_to_train = ["a"]

    # === Finalize and create ===
    trainer_config.update(
        {
            # "env_config": env_config,
            "seed": final_seed,
            "multiagent": {
                "policies": policies,
                "policies_to_train": policies_to_train,
                "policy_mapping_fn": policy_mapping_fun,
            },
            "metrics_smoothing_episodes": 0
            # trainer_config.get("num_workers")* trainer_config.get("num_envs_per_worker"),
        }
    )

    def logger_creator(config):
        return NoopLogger({}, "/tmp")

    register_env("ai-economist-external", lambda _: AieExternalMultiAgentEnv(dummy_env))

    ppo_trainer = PPOTrainer(
        env="ai-economist-external",
        config=trainer_config,
        logger_creator=logger_creator,
    )

    return ppo_trainer


if __name__ == "__main__":
    run_dir, run_config = process_args()
    trainer = build_trainer(run_config)

    dense_log_frequency = run_config["env"].get("dense_log_frequency", 0)
    ckpt_frequency = run_config["general"].get("ckpt_frequency_steps", 0)
    num_parallel_episodes_done = 0
    logger.info("training")
    while num_parallel_episodes_done < run_config["general"]["episodes"]:
        logger.info(f"training: {num_parallel_episodes_done}")

        # Training
        result = trainer.train()

        # === Counters++ ===
        num_parallel_episodes_done = result["episodes_total"]
        global_step = result["timesteps_total"]
        curr_iter = result["training_iteration"]

        if curr_iter == 1 or result["episodes_this_iter"] > 0:
            logger.info(pretty_print(result))

    ray.shutdown()  # shutdown Ray after use
