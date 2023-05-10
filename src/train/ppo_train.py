"""
Defines the configuration for training with offline learning.

Supports PPO.
"""
import logging
import multiprocessing
import os
import time
from datetime import datetime
import toml

import torch
from torch.multiprocessing import Pipe, Process
from tqdm import tqdm

from src.common import EmptyModel, test_mapping
from src.train.ppo import PpoPolicy, RolloutWorker
from src.train.ppo.utils import load_batch, delete_batch
from src.common.rollout_buffer import RolloutBuffer


def run_rollout_worker(conn, worker: RolloutWorker):
    while True:
        weights = conn.recv()
        worker.set_weights(weights=weights)
        worker.batch()
        # Bad way of using semaphores/signals
        conn.send(1)
        worker.log_rewards()


class PpoTrainConfig:
    """
    Endpoint to setup PPO's training configuration.

    Args:
        step:
        seed:

        env:
        rollout_fragment_length:


        k_epochs:
        eps_clip:
        gamma:
        device:
        learning_rate:
        num_workers:
        mapping_function:
        mapped_agents:
        c1:
        c2:
    """

    def __init__(
        self,
        mapping_function,
        env,
        batch_size: int = 6000,
        rollout_fragment_length: int = 200,
        step: int = 100,
        seed: int = 1,
        k_epochs: int = 16,
        eps_clip: int = 10,
        gamma: float = 0.998,
        device: str = "cpu",
        learning_rate: float = 0.0003,
        num_workers: int = 12,
        mapped_agents: dict = {"a": True, "p": False},
        _c1: float = 0.05,
        _c2: float = 0.025,
    ):
        ## Save variables
        self.mapping_function = mapping_function
        self.rollout_fragment_length = rollout_fragment_length
        self.batch_size = batch_size
        self.step = step
        self.seed = seed
        self.k_epochs = batch_size//rollout_fragment_length # k_epochs
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.device = device
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.mapped_agents = mapped_agents
        self._c1 = _c1
        self._c2 = _c2

        self.policy_keys = mapped_agents.keys()

        ## Seeding
        torch.manual_seed(seed)

        ## Validate config
        self.validate_config(env=env)

        ## Determine which phase we are doing
        self.phase = (
            "P1"
            if (
                isinstance(self.mapped_agents["p"], bool)
                and self.mapped_agents["p"] is False
            )
            else "P2"
        )

        ## Create directory
        self.experiment_name = self.setup_logs_and_dirs()

        ## Build trainer
        trainer_config = self.setup_config(env)

        self.build_workers(trainer_config, env, seed)
        logging.debug("Ready! -- Everything built!")

    def train(self):
        """
        Simple train function that does training for `self.step` steps,
        saves all models and closes workers.
        """
        for _ in tqdm(range(self.step)):
            self.train_one_step()

        self.save_models()
        self.close_workers()

    def maybe_load_models(self):
        """
        Load models from another experiment.
        IT checks if the mapped agent is not bool, if so it
        uses its value as the path of the experiment.
        """
        models_to_load = {}
        if not isinstance(self.mapped_agents["a"], bool):
            models_to_load["a"] = self.mapped_agents["a"]

        if not isinstance(self.mapped_agents["p"], bool):
            models_to_load["p"] = self.mapped_agents["p"]

        self.learn_worker.load_models(models_to_load)

        logging.info(f"Loaded models {models_to_load.keys()}")

    def build_workers(self, config, env, seed):
        """
        Builds workers to learn and create the batch.
        """
        obs = env.reset()
        actor_keys = obs.keys()

        self.learn_worker = RolloutWorker(
            rollout_fragment_length=0,
            batch_iterations=0,
            policies_config=config,
            actor_keys=actor_keys,
            mapping_function=self.mapping_function(),
            env=env,
            seed=seed,
            experiment_name=self.experiment_name,
        )
        self.maybe_load_models()

        self.memory = RolloutBuffer(obs, self.mapping_function())
        self.memory.clear()

        # Multi-processing
        # Spawn secondary workers used in batching
        self.pipes = [Pipe() for _ in range(self.num_workers)]

        # Calculate batch iterations distribution
        if self.batch_size % self.rollout_fragment_length != 0:
            raise ValueError(f"train_batch_size % rollout_fragment_length must be == 0")

        batch_iterations = self.batch_size // self.rollout_fragment_length
        iterations_per_worker = batch_iterations // self.num_workers
        remaining_iterations = batch_iterations - (
            iterations_per_worker * self.num_workers
        )

        self.workers = []
        self.workers_id = []
        for _id in range(self.num_workers):
            # Get pipe connection
            parent_conn, child_conn = self.pipes[_id]

            # Calculate worker_iterations
            if remaining_iterations > 0:
                worker_iterations = iterations_per_worker + 1
                remaining_iterations -= 1
            else:
                worker_iterations = iterations_per_worker

            worker = RolloutWorker(
                rollout_fragment_length=self.rollout_fragment_length,
                batch_iterations=worker_iterations,
                policies_config=config,
                actor_keys=actor_keys,
                mapping_function=self.mapping_function(),
                env=env,
                _id=_id,
                seed=seed,
                experiment_name=self.experiment_name,
            )

            self.workers_id.append(_id)

            p = Process(
                target=run_rollout_worker,
                name=f"RolloutWorker-{_id}",
                args=(child_conn, worker),
            )

            self.workers.append(p)
            p.start()
            parent_conn.send(self.learn_worker.get_weights())

        logging.info("Rollout workers built!")

    def train_one_step(self):
        """
        Train all policies.
        It creates a batch of size = `self.batch_size`, then
        this RolloutBuffer is splitted between each policy following
        `self.policy_mapping_fun` and trained respectivly to the
        corrisponding policy.
        """
        # Get batches and create a single "big" batch
        _ = [pipe[0].recv() for pipe in self.pipes]

        # Open all files in a list of `file`
        for worker_id in self.workers_id:
            rollout = load_batch(worker_id=worker_id)
            self.memory.extend(rollout)

        tensored_memory = self.memory.to_tensor()

        self.learn_worker.learn(memory=tensored_memory)

        # Send updated policy to all rollout workers
        for pipe in self.pipes:
            pipe[0].send(self.learn_worker.get_weights())

        # Clear memory from used batch
        self.memory.clear()

    def close_workers(self):
        """
        Kill and clear all workers.
        """
        delete_batch(self.workers_id)

        for worker in self.workers:
            worker.kill()

    def setup_config(self, env):
        """
        Bad implementation to manage policies.
        """
        env.reset()

        config = {}
        config["a"] = {
            "policy": PpoPolicy,
            "observation_space": env.observation_space,
            "action_space": 50,
            "k_epochs": self.k_epochs,
            "eps_clip": self.eps_clip,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "c1": self._c1,
            "c2": self._c2,
            "device": self.device,
            "name": "a",
        }
        if self.mapped_agents["p"] is True:
            config["p"] = {
                "policy": PpoPolicy,
                "observation_space": env.observation_space_pl,
                "action_space": 154,
                "k_epochs": self.k_epochs,
                "eps_clip": self.eps_clip,
                "gamma": self.gamma,
                "learning_rate": self.learning_rate,
                "c1": self._c1,
                "c2": self._c2,
                "device": self.device,
                "name": "p",
            }
        else:
            config["p"] = {
                "policy": EmptyModel,
                "observation_space": env.observation_space_pl,
                "action_space": 7,
            }

        return config

    def validate_config(self, env):
        """
        Validate PPO's config.

        Raise:
            ValueError if a specific parameter is not set correctly.
        """
        test_mapping(
            mapping_function=self.mapping_function,
            mapped_agents=self.mapped_agents,
            env=env,
        )

        if self.step < 0:
            raise ValueError("'step' must be > 0!")

        if not isinstance(self.seed, int):
            raise ValueError("'seed' muse be integer!")

        if self.k_epochs < 0:
            raise ValueError("'k_epochs' must be > 0!")

        if self.eps_clip < 0:
            raise ValueError("'eps_clip' must be > 0!")

        if self.gamma < 0 or self.gamma > 1:
            raise ValueError("'gamma' must be between (0,1).")

        if self.device != "cpu":
            # raise ValueError()
            logging.warning(
                "The only available 'device' at the moment is 'cpu'. Redirecting everything to 'cpu'!"
            )
            self.device = "cpu"

        if self.learning_rate < 0 and self.learning_rate > 1:
            raise ValueError("'learning_rate' must be between (0,1).")

        if self.num_workers < 0:
            raise ValueError("'num_workers' must be > 0 and < max_cpus.")

        if self.num_workers != multiprocessing.cpu_count():
            logging.warning(
                "You should use %s workers instead of %s.",
                multiprocessing.cpu_count(),
                self.num_workers,
            )

        logging.info("Configuration validated: OK")
        return

    def setup_logs_and_dirs(self):
        """
        Creates this experiment directory and a log file.
        """

        date = datetime.today().strftime("%d-%m-%Y")
        experiment_name = f"PPO_{self.phase}_{date}_{int(time.time())}_{self.step}"
        path = f"experiments/{experiment_name}"

        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path + "/logs")
            os.makedirs(path + "/models")
            os.makedirs(path + "/plots")

        # Create config.txt log file
        with open(path + "/config.toml", "w") as config_file:
            config_dict = {
                "common": {
                    "algorithm_name": "PPO",
                    "phase": self.phase,
                    "step": self.step,
                    "seed": self.seed,
                    "device": self.device,
                    "mapped_agents": self.mapped_agents,
                },
                "algorithm_specific": {
                    "rollout_fragment_length": self.rollout_fragment_length,
                    "batch_size": self.batch_size,
                    "k_epochs": self.k_epochs,
                    "eps_clip": self.eps_clip,
                    "gamma": self.gamma,
                    "learning_rate": self.learning_rate,
                    "num_workers": self.num_workers,
                    "c1": self._c1,
                    "c2": self._c2,
                },
            }
            toml.dump(config_dict, config_file)

        logging.info("Directories created")
        return experiment_name

    def save_models(self):
        self.learn_worker.save_models()
        logging.info("Models saved")

    def load_models(self):
        self.learn_worker.load_models()
