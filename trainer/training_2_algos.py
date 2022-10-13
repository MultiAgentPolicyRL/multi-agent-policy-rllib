# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

# Documentation Google Style
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# How it works
# https://bair.berkeley.edu/blog/2018/12/12/rllib/
# https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_two_trainers.py
# Edited by Ettore Saggiorato - GitHub@Sa1g

from ray.rllib.agents.ppo import PPOTrainer
from pkg_resources import get_distribution
from ray.tune.registry import register_env
from ray.tune.logger import NoopLogger, pretty_print
from env_wrapper import RLlibEnvWrapper
import yaml
from tf_models import KerasConvLSTM  # used by config.yaml
from utils import remote, saving
import ray
import time
import sys
import os
import logging
import argparse
from random import seed

import warnings
warnings.filterwarnings("ignore")

# For ray[rllib]==0.8.3, 0.8.4, 0.8.5

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


"""
Working Emojis
✅ - (Mine) Done, don't touch anymore
🚫 - Dont't touch
🟢 - Mine, done, not perfect yet
🟡🚸 - Mine, WORK IN PROGESS
🟠 - Mine, improve
🔴 - Mine, it's trash, work ok it 
🟪 - Original Code, untouched yet
"""

# 🟢 Untouched - added docs


def process_args():
    """
    Processes arguments, checks for correct directory reference and config.yaml file and redis password.

    Args:
        None

    Raises:
        AssertionError: run_dir is dir
        AssertionError: config.yaml in run_dir

    Returns:
        ``run_directory`` (path)
        ``run_configuration`` (yaml)
        ``redis_pwd`` (password, string)
        ``cluster`` (boolean)
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to the directory for this run.",
        default="phase1",
    )

    parser.add_argument(
        "--pw",
        type=str,
        help="Redis password.",
        default="password",
    )

    parser.add_argument(
        "--ip_address",
        type=str,
        help="Ray ip:port",
        default="",
    )

    parser.add_argument(
        "--cluster",
        type=bool,
        help="If experiment is running on a cluster",
        default=False,
    )

    args = parser.parse_args()
    run_directory = args.run_dir

    config_path = os.path.join(args.run_dir, "config.yaml")
    assert os.path.isdir(args.run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    return run_directory, run_configuration, args.pw, args.ip_address, args.cluster


# 🚸 work in progress
def build_trainer(run_configuration):
    """Finalize the trainer config by combining the sub-configs.

    Args:
        run_configuration (loaded yaml): configuration file loaded by ``process_args``

    Returns:
        ppo_trainer: RLLib PPOTrainer with injected config.

    Todo:
        * most probably modify this function to have a multi-algo trainer
    """
    trainer_config = run_configuration.get("trainer")

    # 🚸 Untouched
    # === Env ===
    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }
    register_env("ai-economist", lambda _: RLlibEnvWrapper(env_config))

    # 🚫 Untouched
    # === Seeding ===
    if trainer_config["seed"] is None:
        try:
            start_seed = int(run_configuration["metadata"]["launch_time"])
        except KeyError:
            start_seed = int(time.time())
    else:
        start_seed = int(trainer_config["seed"])

    final_seed = int(start_seed % (2**16)) * 1000
    logger.info("seed (final): %s", final_seed)

    # === Multiagent Policies ===
    dummy_env = RLlibEnvWrapper(env_config)

    # 🟢 66-97 multi_agent_two_trainers
    # Policy tuples for agent/planner policy types

    policies = {
        "agent_policy": (
            None,
            dummy_env.observation_space,
            dummy_env.action_space,
            run_configuration.get("agent_policy"),
        ),
        "planner_policy": (
            None,
            dummy_env.observation_space_pl,
            dummy_env.action_space_pl,
            run_configuration.get("planner_policy"),
        ),
    }

    # 🟢 - modified original code a little bit
    # 99-105 multi_agent_two_trainers
    # Level 2
    def policy_mapping_fun(i):
        if str(i).isdigit() or i == "a":
            return "agent_policy"
        return "planner_policy"

    # 🟢 lines 105-125
    ppoAgent = PPOTrainer(
        env="ai-economist",
        config={
            "env_config": env_config,
            "seed": final_seed,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fun,
                "policies_to_train": ["agent_policy"],
            },
            "metrics_smoothing_episodes": trainer_config.get("num_workers")
            * trainer_config.get("num_envs_per_worker"),
        },
        # logger_creator=logger_creator
    )

    # 🟡 lines 127-144 - PPO Planner isn't the final scope
    # editing config.yaml and setting for other algos it's possible to use
    # all `ray.rllib.agents` trainers.
    ppoPlanner = PPOTrainer(
        env="ai-economist",
        config={
            "env_config": env_config,
            "seed": final_seed,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fun,
                "policies_to_train": ["planner_policy"],
            },
            "metrics_smoothing_episodes": trainer_config.get("num_workers")
            * trainer_config.get("num_envs_per_worker"),
        },
        # logger_creator=logger_creator
    )

    # def logger_creator(config):
    #     return NoopLogger({}, "/tmp")

    # 🟢
    return ppoAgent, ppoPlanner


# 🟢
def set_up_dirs_and_maybe_restore(
    run_directory, run_configuration, trainerAgent, trainerPlanner
):
    # === Set up Logging & Saving, or Restore ===
    # All model parameters are always specified in the settings YAML.
    # We do NOT overwrite / reload settings from the previous checkpoint dir.
    # 1. For new runs, the only object that will be loaded from the checkpoint dir
    #    are model weights.
    # 2. For crashed and restarted runs, load_snapshot will reload the full state of
    #    the Trainer(s), including metadata, optimizer, and models.
    (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
    ) = saving.fill_out_run_dir(run_directory)

    # If this is a starting from a crashed run, restore the last trainer snapshot
    if restore_from_crashed_run:
        logger.info(
            "ckpt_dir already exists! Planning to restore using latest snapshot from "
            "earlier (crashed) run with the same ckpt_dir %s",
            ckpt_directory,
        )

        at_loads_agents_ok = saving.load_snapshot(
            trainerAgent, run_directory, load_latest=True, suffix="agent"
        )

        at_loads_planner_ok = saving.load_snapshot(
            trainerPlanner, run_directory, load_latest=True, suffix="planner"
        )

        # at this point, we need at least one good ckpt restored
        if not at_loads_agents_ok and not at_loads_planner_ok:
            logger.fatal(
                "restore_from_crashed_run -> restore_run_dir %s, but no good ckpts "
                "found/loaded!",
                run_directory,
            )
            sys.exit()

        # === Trainer-specific counters ===
        # it's the same for Agents and PPO
        training_step_last_ckpt = (
            int(trainerAgent._timesteps_total) if trainerAgent._timesteps_total else 0
        )
        epis_last_ckpt = (
            int(trainerAgent._episodes_total) if trainerAgent._episodes_total else 0
        )

    else:

        logger.info("Not restoring trainer...")
        # === Trainer-specific counters ===
        training_step_last_ckpt = 0
        epis_last_ckpt = 0

        # == For new runs, load only tf checkpoint weights ==

        # Agents
        starting_weights_path_agents = run_configuration["general"].get(
            "restore_tf_weights_agents", ""
        )
        if starting_weights_path_agents:
            logger.info("Restoring agents TF weights...")
            saving.load_tf_model_weights(
                trainerAgent, starting_weights_path_agents)
        else:
            logger.info("Starting with fresh agent TF weights.")

        # Planner
        starting_weights_path_planner = run_configuration["general"].get(
            "restore_tf_weights_planner", ""
        )
        if starting_weights_path_planner:
            logger.info("Restoring planner TF weights...")
            saving.load_tf_model_weights(
                trainerPlanner, starting_weights_path_planner)
        else:
            logger.info("Starting with fresh planner TF weights.")

    return (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
        training_step_last_ckpt,
        epis_last_ckpt,
    )


# 🟢 changed agent_trainer, added planner_trainer and ifPlanner
def maybe_store_dense_log(
    agent_trainer,
    planner_trainer,
    result_dict,
    dense_log_freq,
    dense_log_directory,
    ifPlanner,
):
    if result_dict["episodes_this_iter"] > 0 and dense_log_freq > 0:
        episodes_per_replica = (
            result_dict["episodes_total"] // result_dict["episodes_this_iter"]
        )
        if episodes_per_replica == 1 or (episodes_per_replica % dense_log_freq) == 0:
            log_dir = os.path.join(
                dense_log_directory,
                "logs_{:016d}".format(result_dict["timesteps_total"]),
            )
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)

            saving.write_dense_logs(agent_trainer, log_dir)

            if ifPlanner:
                saving.write_dense_logs(planner_trainer, log_dir, "planner")

            logger.info(">> Wrote dense logs to: %s", log_dir)


# 🟢 changed agent_trainer, added planner_trainer and ifPlanner
def maybe_save(
    agent_trainer,
    planner_trainer,
    result_dict,
    ckpt_freq,
    ckpt_directory,
    trainer_step_last_ckpt,
    ifPlanner,
):
    global_step = result_dict["timesteps_total"]

    # Check if saving this iteration
    if (
        result_dict["episodes_this_iter"] > 0
    ):  # Don't save if midway through an episode.

        if ckpt_freq > 0:
            if global_step - trainer_step_last_ckpt >= ckpt_freq:
                saving.save_snapshot(
                    agent_trainer, ckpt_directory, suffix="agent")
                saving.save_tf_model_weights(
                    agent_trainer, ckpt_directory, global_step, suffix="agent"
                )

                if ifPlanner:
                    saving.save_snapshot(
                        planner_trainer, ckpt_directory, suffix="planner"
                    )
                    saving.save_tf_model_weights(
                        planner_trainer, ckpt_directory, global_step, suffix="planner"
                    )

                trainer_step_last_ckpt = int(global_step)

                logger.info("Checkpoint saved @ step %d", global_step)

    return trainer_step_last_ckpt


if __name__ == "__main__":

    # ===================
    # === Start setup ===
    # ===================

    # Process the args
    run_dir, run_config, redis_pwd, ip_address, cluster = process_args()

    # if experiment is run on a cluster the experiment launcher is going to connect to a remote ray_core with
    # address={ip_address} and redis_password=redis_pwd
    if (cluster):
        print("using the cluster")
        ray.init(log_to_driver=False, address=(
            f"{ip_address}"), redis_password=redis_pwd)
    else:
        print("training locally")
        ray.init(log_to_driver=False)

    # Create a trainer object
    trainerAgents, trainerPlanner = build_trainer(run_config)

    (
        # 🔴 - simple training without any saving/checkpoint/tooling
        # the only weight we care to save during Phase1 is Agent's bc it will
        # be loaded during Phase2 (as paper, to do not discourage exploration.)
        # 🚫 don't touch - backup
        # for i in range(run_config["general"]["episodes"]):
        #     print(f"== Iteration {i} ==")
        #     # Improve trainerAgents policy's
        #     print("-- PPO Agents --")
        #     result_ppo_agents = trainerAgents.train()
        #     # print(f"{result_ppo_agents['episode_reward_max']}, {result_ppo_agents['episode_reward_min']}, {result_ppo_agents['episode_reward_mean']}, {result_ppo_agents['episode_len_mean']}, {result_ppo_agents['episodes_this_iter']}")
        #     print(pretty_print(result_ppo_agents))
        #     if run_config["general"]["train_planner"]:
        #         # Improve trainerPlanner policy's
        #         print("-- PPO Planner --")
        #         result_ppo_planner = trainerPlanner.train()
        #         # print(f"{result_ppo_planner['episode_reward_max']}, {result_ppo_planner['episode_reward_min']}, {result_ppo_planner['episode_reward_mean']}, {result_ppo_planner['episode_len_mean']}, {result_ppo_planner['episodes_this_iter']}")
        #         print(pretty_print(result_ppo_planner))
        #         # Swap weights to synchronize
        #         trainerAgents.set_weights(
        #             trainerPlanner.get_weights(["planner_policy"]))
        #         trainerPlanner.set_weights(trainerAgents.get_weights(["agent_policy"]))
    )
    # Set up directories for logging and saving. Restore if this has already been
    # done (indicating that we're restarting a crashed run). Or, if appropriate,
    # load in starting model weights for the agent and/or planner.
    # 🟠 - fix trainer
    (
        dense_log_dir,
        ckpt_dir,
        restore_from_crashed_run,
        step_last_ckpt,
        num_parallel_episodes_done,
    ) = set_up_dirs_and_maybe_restore(
        run_dir, run_config, trainerAgents, trainerPlanner
    )

    # # ======================
    # # === Start training ===
    # # ======================

    dense_log_frequency = run_config["env"].get("dense_log_frequency", 0)
    ckpt_frequency = run_config["general"].get("ckpt_frequency_steps", 0)
    global_step = int(step_last_ckpt)
    ifPlanner = run_config["general"]["train_planner"]

    print("Training")
    while num_parallel_episodes_done < run_config["general"]["episodes"]:
        # === Training ===
        """
        Should we use tune.run for training or rllib training?
        """
        # Improve trainerAgents policy's
        print(f"-- PPO Agents -- Steps done: {num_parallel_episodes_done}")

        result_ppo_agents = trainerAgents.train()

        # print(f"{result_ppo_agents['episode_reward_max']}, {result_ppo_agents['episode_reward_min']}, {result_ppo_agents['episode_reward_mean']}, {result_ppo_agents['episode_len_mean']}, {result_ppo_agents['episodes_this_iter']}")
        # print(pretty_print(result_ppo_agents))

        # train Agents and Planner
        if ifPlanner:
            # Improve trainerPlanner policy's
            print("-- PPO Planner --")
            result_ppo_planner = trainerPlanner.train()
            # print(f"{result_ppo_planner['episode_reward_max']}, {result_ppo_planner['episode_reward_min']}, {result_ppo_planner['episode_reward_mean']}, {result_ppo_planner['episode_len_mean']}, {result_ppo_planner['episodes_this_iter']}")
            # print(pretty_print(result_ppo_planner))

            # Swap weights to synchronize
        trainerAgents.set_weights(
            trainerPlanner.get_weights(["planner_policy"]))

        trainerPlanner.set_weights(
            trainerAgents.get_weights(["agent_policy"]))

        # === Counters++ ===
        # episodes_total, timesteps_total, training_iteration is the same for Agents and Planner
        num_parallel_episodes_done = result_ppo_agents["episodes_total"]
        global_step = result_ppo_agents["timesteps_total"]
        curr_iter = result_ppo_agents["training_iteration"]

        # === Logging ===
        # 🟠 add planner infos (Idk if timesteps_this_iter is in the Decision Tree algo)
        logger.info(
            "Iter %d: steps this-iter %d total %d -> %d/%d episodes done",
            curr_iter,
            result_ppo_agents["timesteps_this_iter"],
            global_step,
            num_parallel_episodes_done,
            run_config["general"]["episodes"],
        )

        if curr_iter == 1 or result_ppo_agents["episodes_this_iter"] > 0:
            logger.info(pretty_print(result_ppo_agents))

            if ifPlanner:
                logger.info(pretty_print(result_ppo_planner))

        # === Saez logic ===
        # saez label is not in config.yaml, nor for phase1, nor phase2. So it's not needed.

        # === Dense logging ===
        maybe_store_dense_log(
            trainerAgents, 
            trainerPlanner,
            result_ppo_agents,
            dense_log_frequency,
            dense_log_dir,
            ifPlanner,
        )

        # === Saving ===
        # Saving MUST be done after weights sync!
        step_last_ckpt = maybe_save(
            trainerAgents,
            trainerPlanner,
            result_ppo_agents,
            ckpt_frequency,
            ckpt_dir,
            step_last_ckpt,
            ifPlanner,
        )

    # === Finish up ===
    logger.info("Completing! Saving final snapshot...\n\n")

    saving.save_snapshot(trainerAgents, ckpt_dir, suffix="agent")
    saving.save_tf_model_weights(
        trainerAgents, ckpt_dir, global_step, suffix="agent")

    if ifPlanner:
        saving.save_snapshot(trainerPlanner, ckpt_dir, suffix="planner")
        saving.save_tf_model_weights(
            trainerPlanner, ckpt_dir, global_step, suffix="planner"
        )

    logger.info("Final snapshot saved! All done.")
    ray.shutdown()  # shutdown Ray after use
