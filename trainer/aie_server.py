#!/usr/bin/env python
# https://raw.githubusercontent.com/ray-project/ray/releases/0.8.4/rllib/examples/serving/cartpole_server.py
"""
In two separate shells run:
    $ python aie_server.py --run=[PPO|DQN]
    $ python aie_client.py --inference-mode=local|remote
"""

import argparse
import os
import sys
import time
import logging

import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print, NoopLogger
from env_wrapper import RLlibEnvWrapper
from tf_models import KerasConvLSTM  # used by config.yaml


from regex import D
import yaml

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)

SERVER_ADDRESS = "localhost"
SERVER_PORT_AGENT = 9900
SERVER_PORT_PLANNER = 9911
CHECKPOINT_FILE = "last_checkpoint_{}.out"


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="PPO")
    parser.add_argument("--run-dir", type=str,
                        help="Path to the directory for this run")

    args = parser.parse_args()
    algo = args.run
    run_directory = args.run_dir

    config_path = os.path.join(args.run_dir, "config.yaml")
    assert os.path.isdir(args.run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        run_config = yaml.safe_load(f)

    return algo, run_directory, run_config


def build_trainer(run_config, algo):
    trainer_config = run_config.get("trainer")

    # Env setup
    env_config = {
        "env_config_dict": run_config.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }

    register_env("ai-economist", lambda _: RLlibEnvWrapper(env_config))

    # Seeding
    if trainer_config["seed"] is None:
        try:
            start_seed = int(run_config["metadata"]["launch_time"])
        except KeyError:
            start_seed = int(time.time())
    else:
        start_seed = int(trainer_config["seed"])

    final_seed = int(start_seed % (2**16)) * 1000
    logger.info("seed (final): %s", final_seed)

    # Policies
    dummy_env = RLlibEnvWrapper(env_config)
    if (algo == "PPO"):
        policies = {
            "ppo_agent_policy": (
                None,
                dummy_env.observation_space,
                dummy_env.action_space,
                run_config.get("ppo_agent_policy"),
            ),
            "ppo_planner_policy": (
                None,
                dummy_env.observation_space_pl,
                dummy_env.action_space_pl,
                run_config.get("ppo_planner_policy"),
            ),
        }
    elif (algo == "DQN"):
        NotImplementedError("DQN not implemented at the moment")
    else:
        raise ValueError("--run must be DQN or PPO")

    # Level 2
    def policy_mapping_fun(i):
        if str(i).isdigit() or i == "a":
            return "ppo_agent_policy"
        return "ppo_planner_policy"

    def logger_creator(config):
        return NoopLogger({}, "/tmp")

    if (trainer_config.get("num_workers") == None):
        metrics_smoothing_episodes = trainer_config.get("num_envs_per_worker")
    elif (trainer_config.get("num_envs_per_worker") == None):
        metrics_smoothing_episodes = trainer_config.get("num_workers")
    else:
        metrics_smoothing_episodes = trainer_config.get(
            "num_workers") * trainer_config.get("num_envs_per_worker")

    trainerAgentConfig = trainer_config
    trainerPlannerConfig = trainer_config

    trainerAgentConfig.update({
        # Use the connector server to generate experiences.
        "input": (
            lambda ioctx: PolicyServerInput(
                ioctx, SERVER_ADDRESS, SERVER_PORT_AGENT)
        ),
        # TODO how to have multiple remote workers?
        # TODO use num_workers from config.yaml
        # Use a single worker process to run the server.
        # "num_workers": 0,
        # Disable OPE, since the rollouts are coming from online clients.
        "input_evaluation": [],
        "env_config": env_config,
        "seed": final_seed,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fun,
            "policies_to_train": ["ppo_agent_policy"],
        },
        "metrics_smoothing_episodes": metrics_smoothing_episodes,
    })

    trainerPlannerConfig.update({
        # Use the connector server to generate experiences.
        "input": (
            lambda ioctx: PolicyServerInput(
                ioctx, SERVER_ADDRESS, SERVER_PORT_PLANNER)
        ),
        # TODO how to have multiple remote workers?
        # TODO use num_workers from config.yaml
        # Use a single worker process to run the server.
        # "num_workers": 0,
        # Disable OPE, since the rollouts are coming from online clients.
        "input_evaluation": [],
        "env_config": env_config,
        "seed": final_seed,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fun,
            "policies_to_train": ["ppo_planner_policy"],
        },
        "metrics_smoothing_episodes": metrics_smoothing_episodes,
    })

    if (algo == "PPO"):
        TrainerAgent: PPOTrainer = PPOTrainer(
            env="ai-economist",
            config=trainerAgentConfig,
            logger_creator=logger_creator
        )

        """ 🟡 PPO Planner isn't the final scope
        editing config.yaml and setting for other algos it's possible to use
        all `ray.rllib.agents` trainers."""
        TrainerPlanner: PPOTrainer = PPOTrainer(
            env="ai-economist",
            config=trainerPlannerConfig,
            logger_creator=logger_creator
        )
    elif(algo == "DQN"):
        NotImplementedError("DQN not implemented at the moment")
    else:
        raise ValueError("--run must be DQN or PPO")
    TrainerPlanner = None
    return TrainerAgent, TrainerPlanner


if __name__ == "__main__":
    algo, run_directory, run_config = process_args()

    ray.init()

    env = "CartPole-v0"
    trainerAgents, trainerPlanner = build_trainer(run_config, algo)
    # skipped restore and logging and saving

    """
    # if algo == "DQN":
    #     NotImplementedError("DQN not available at the moment")
    #     # Example of using DQN (supports off-policy actions).
    #     trainer = DQNTrainer(
    #         env=env,
    #         config={
    #             # Use the connector server to generate experiences.
    #             "input": (
    #                 lambda ioctx: PolicyServerInput( \
    #                     ioctx, SERVER_ADDRESS, SERVER_PORT)
    #             ),
    #             # Use a single worker process to run the server.
    #             "num_workers": 0,
    #             # Disable OPE, since the rollouts are coming from online clients.
    #             "input_evaluation": [],
    #             "exploration_config": {
    #                 "type": "EpsilonGreedy",
    #                 "initial_epsilon": 1.0,
    #                 "final_epsilon": 0.02,
    #                 "epsilon_timesteps": 1000,
    #             },
    #             "learning_starts": 100,
    #             "timesteps_per_iteration": 200,
    #             "log_level": "INFO",
    #         })
    # elif algo == "PPO":
    #     # Example of using PPO (does NOT support off-policy actions).
    #     trainer = PPOTrainer(
    #         env=env,

    #         config={
    #             # Use the connector server to generate experiences.
    #             "input": (
    #                 lambda ioctx: PolicyServerInput( \
    #                     ioctx, SERVER_ADDRESS, SERVER_PORT)
    #             ),
    #             # Use a single worker process to run the server.
    #             "num_workers": 0,
    #             # Disable OPE, since the rollouts are coming from online clients.
    #             "input_evaluation": [],
    #             "sample_batch_size": 1000,
    #             "train_batch_size": 4000,
    #         }
    #     )
    # else:
    #     raise ValueError("--run must be DQN or PPO")
"""

    # checkpoint_path = CHECKPOINT_FILE.format(algo)

    # # Attempt to restore from checkpoint if possible.
    # if os.path.exists(checkpoint_path):
    #     checkpoint_path = open(checkpoint_path).read()
    #     print("Restoring from checkpoint path", checkpoint_path)
    #     trainer.restore(checkpoint_path)

    dense_log_frequency = run_config["env"].get("dense_log_frequency", 0)
    ckpt_frequency = run_config["general"].get("ckpt_frequency_steps", 0)
    # global_step = int(step_last_ckpt)
    ifPlanner = run_config["general"]["train_planner"]

    # NOTE: TMP
    num_parallel_episodes_done = 0

    # Serving and training loop
    logger.info("Training")
    while num_parallel_episodes_done < run_config["general"]["episodes"]:
        # Improve trainerAgents policy's
        result_ppo_agents = trainerAgents.train()

        if ifPlanner:
            # Improve trainerPlanner policy's
            result_ppo_planner = trainerPlanner.train()

            # # Swap weights to synchronize
            trainerAgents.set_weights(
                trainerPlanner.get_weights(["planner_policy"]))
            trainerPlanner.set_weights(
                trainerAgents.get_weights(["agent_policy"]))

        # === Counters++ ===
        # episodes_total, timesteps_total, training_iteration is the same for Agents and Planner
        num_parallel_episodes_done = result_ppo_agents["episodes_total"]
        # global_step = result_ppo_agents["timesteps_total"]
        curr_iter = result_ppo_agents["training_iteration"]

        if curr_iter == 1 or result_ppo_agents["episodes_this_iter"] > 0:
            logger.info(pretty_print(result_ppo_agents))

            if ifPlanner:
                logger.info(pretty_print(result_ppo_planner))
        # === Saez logic ===
        # saez label is not in config.yaml, nor for phase1, nor phase2. So it's not needed.

        # TODO missing parts
        # === Dense logging ===
        # === Saving ===
        # Saving MUST be done after weights sync! -> it's saving weights!

        # checkpoint = trainer.save()
        # print("Last checkpoint", checkpoint)
        # with open(checkpoint_path, "w") as f:
        #     f.write(checkpoint)

    # logger.info("Done, shut down ray and exit")
    # ray.shutdown()  # shutdown Ray after use
