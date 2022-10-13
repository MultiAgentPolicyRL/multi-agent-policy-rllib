# Copyright (c) 2022, Ettore Saggiorato - GitHub@Sa1g
# Documentation Google Style
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
# How it works
# https://bair.berkeley.edu/blog/2018/12/12/rllib/
# https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_two_trainers.py

import ray
import time
import sys
import os
import yaml
import logging
import argparse

from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune.logger import NoopLogger, pretty_print
from env_wrapper import RLlibEnvWrapper
from tf_models import KerasConvLSTM  # used by config.yaml
from utils import saving
from utils import dirs_restore_logs_save

import warnings
warnings.filterwarnings("ignore")

# For ray[rllib]==0.8.3, 0.8.4, 0.8.5

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

# ✅🚫🟢🟡🚸🟠🔴🟪

# ✅
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

    parser.add_argument("--run-dir", type=str,
                        help="Path to the directory for this run.", default="phase1",)

    parser.add_argument(
        "--pw", type=str, help="Redis password, used only when clustering", default="password",)

    parser.add_argument("--ip_address", type=str,
                        help="Ray ip:port, used only when clustering", default="",)

    parser.add_argument("--cluster", type=bool,
                        help="If experiment is running on a cluster set to `True`, otherwise don't use", default=False, )

    args = parser.parse_args()
    run_directory = args.run_dir

    config_path = os.path.join(args.run_dir, "config.yaml")
    assert os.path.isdir(args.run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    return run_directory, run_configuration, args.pw, args.ip_address, args.cluster

# 🟢 
def build_trainer(run_configuration):
    """Finalize the trainer config by combining the sub-configs. It makes multi agent two trainers available.  

    Like: https://github.com/ray-project/ray/blob/master/rllib/examples/multi_agent_two_trainers.py

    Args:
        ``run_configuration`` (loaded yaml): configuration file loaded by ``process_args``

    Returns:
        ``ppoAgent``: RLLib PPOTrainer with injected config for environment's Agents
        ``ppoPlanner``: RLLib PPOTrainer with injected config for environment's Social Planner
    """
    trainer_config = run_configuration.get("trainer")

    # === Env ===
    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }

    register_env("ai-economist", lambda _: RLlibEnvWrapper(env_config))

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

    # Level 2
    def policy_mapping_fun(i):
        if str(i).isdigit() or i == "a":
            return "agent_policy"
        return "planner_policy"

    def logger_creator(config):
        return NoopLogger({}, "/tmp")

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
        logger_creator=logger_creator
    )

    """ 🟡 PPO Planner isn't the final scope
    editing config.yaml and setting for other algos it's possible to use
    all `ray.rllib.agents` trainers."""
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
        logger_creator=logger_creator
    )

    return ppoAgent, ppoPlanner



"""
If experiment is run on a cluster the experiment launcher is going to connect to a remote 
ray_core with ```address=ip_address``` and ```redis_password=redis_pwd```
"""
if __name__ == "__main__":
    # Process the args
    run_dir, run_config, redis_pwd, ip_address, cluster = process_args()

    if (cluster):
        logger.info("using the cluster")
        ray.init(log_to_driver=False, address=(
            f"{ip_address}"), redis_password=redis_pwd)
    else:
        logger.info("training locally")
        ray.init(log_to_driver=False)

    # Create trainer objects
    trainerAgents, trainerPlanner = build_trainer(run_config)

    # Set up directories for logging and saving. Restore if this has already been
    # done (indicating that we're restarting a crashed run). Or, if appropriate,
    # load in starting model weights for the agent and/or planner.
    (
        dense_log_dir,
        ckpt_dir,
        restore_from_crashed_run,
        step_last_ckpt,
        num_parallel_episodes_done,
    ) = dirs_restore_logs_save.set_up_dirs_and_maybe_restore(
        run_dir, run_config, trainerAgents, trainerPlanner
    )

    # # ======================
    # # === Start training ===
    # # ======================

    dense_log_frequency = run_config["env"].get("dense_log_frequency", 0)
    ckpt_frequency = run_config["general"].get("ckpt_frequency_steps", 0)
    global_step = int(step_last_ckpt)
    ifPlanner = run_config["general"]["train_planner"]

    logger.info("Training")
    while num_parallel_episodes_done < run_config["general"]["episodes"]:
        # === Training ===
        """
        Should we use tune.run for training or rllib training?
        """
        # Improve trainerAgents policy's
        result_ppo_agents = trainerAgents.train()

        # train Agents and Planner
        if ifPlanner:
            # Improve trainerPlanner policy's
            result_ppo_planner = trainerPlanner.train()

        # Swap weights to synchronize
        trainerAgents.set_weights(trainerPlanner.get_weights(["planner_policy"]))
        trainerPlanner.set_weights(trainerAgents.get_weights(["agent_policy"]))

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
        dirs_restore_logs_save.maybe_store_dense_log(
            trainerAgents,
            trainerPlanner,
            result_ppo_agents,
            dense_log_frequency,
            dense_log_dir,
            ifPlanner,
        )

        # === Saving ===
        # Saving MUST be done after weights sync! -> it's saving weights!
        step_last_ckpt = dirs_restore_logs_save.maybe_save(
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
