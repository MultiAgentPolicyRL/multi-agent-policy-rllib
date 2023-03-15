"""
Defines the configuration for training with offline learning.

Supports PPO.
"""
from datetime import datetime
import logging
import multiprocessing
import os
import time
from src.common import test_mapping


# FIXME leggi qui.
"""
questi "train" sono i manger di setup e training.

In pratica creano l'environment, gestiscono il multi-agente,
gestiscono e creano i worker, assemblano la batch e fanno l'apprendimento.

alla fine dell'apprendimento salvano i modelli.

il caricamento dei modelli e' fatto in fase di inizializzazione se
il flag non e' booleano.
"""

class PpoTrainConfig:
    """
    Endpoint to setup PPO's training configuration.

    Args:
        step:
        seed:

        k_epochs:
        eps_clip:
        gamma:
        device:
        learning_rate:
        num_workers:
        mapping_function:
        mapped_agents:
    """

    def __init__(
        self,
        mapping_function,
        env,
        step: int = 100,
        seed: int = 1,
        k_epochs: int = 16,
        eps_clip: int = 10,
        gamma: float = 0.998,
        device: str = "cpu",
        learning_rate: float = 0.0003,
        num_workers: int = 12,
        mapped_agents: dict = {"a": True, "p": False},
    ):
        ## Save variables
        self.mapping_function = mapping_function
        self.step = step
        self.seed = seed
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.device = device
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.mapped_agents = mapped_agents

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
        self.setup_logs_and_dirs()

        ## Build trainer
        build_workers()
        maybe_load_models()


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

        return

    def setup_logs_and_dirs(self):
        """
        Creates this experiment directory and a log file.
        """
        
        date = datetime.today().strftime("%d-%m-%Y")

        path = f"experiments/PPO_{self.phase}_{date}_{int(time.time())}_{self.step}"

        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path + "/logs")

        # Create config.txt log file
        with open(path + "/config.txt", "w") as config_file:
            config_file.write(f"algorithm: PPO\nstep: {self.step}\nseed: {self.seed}\n")
            config_file.write(
                f"k_epochs: {self.k_epochs}\neps_clip: {self.eps_clip}\ngamma: {self.gamma}\ndevice: {self.device}\nlearning_rate: {self.learning_rate}\nnum_workers: {self.num_workers}\nmapped_agents: {self.mapped_agents}\n"
            )
