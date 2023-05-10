"""
TODO
"""
import logging
import os
import shutil
import json
import pickle
import time
import random
from datetime import datetime
import numpy as np

import toml
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.interact import plotting

from src.common import EmptyModel, test_mapping
from src.train.dt_ql_train import DtTrainConfig
from src.train.ppo_train import PpoTrainConfig
from src.train.ppo_dt import PPODtTrainConfig


class InteractConfig:
    """
    Endpoint to setup interact configuration.

    Args:
        env
        trainer: it can be one between DtTrainConfig and PpoTrainConfig
        steps: how many steps to step the environment with these pretrained models
        config: TODO
        mapped_agents: the same as in PpoTrainConfig
    """

    def __init__(
        self,
        mapping_function,
        env,
        trainer,
        config: dict,
        mapped_agents: dict,
        device: str = "cpu",
        seed: int = 1,
    ):
        self.mapping_function = mapping_function
        self.env = env
        self.device = device
        self.trainer = trainer
        self.seed = seed
        self.config = config
        self.mapped_agents = mapped_agents

        self.validate_config(env=env)
        self.setup_logs_and_dirs()

        # Build correct stepper (build and load specific models)
        dense_logs = self.run_stepper()

        (fig0, fig1, fig2), incomes, endows, c_trades, all_builds = plotting.breakdown(
            dense_logs
        )

        fig0.savefig(fname=os.path.join(self.path, "plots", "Global.png"))

        fig1.savefig(fname=os.path.join(self.path, "plots", "Trend.png"))

        fig2.savefig(fname=os.path.join(self.path, "plots", "Movements.png"))

        with open(
            os.path.join(self.path, "logs", "incomes.pkl"), "+wb"
        ) as incomes_file:
            pickle.dump(incomes, incomes_file)

        with open(os.path.join(self.path, "logs", "endows.pkl"), "+wb") as endows_file:
            pickle.dump(endows, endows_file)

        with open(
            os.path.join(self.path, "logs", "c_trades.pkl"), "+wb"
        ) as c_trades_file:
            pickle.dump(c_trades, c_trades_file)

        with open(
            os.path.join(self.path, "logs", "all_builds.pkl"), "+wb"
        ) as all_builds_file:
            pickle.dump(all_builds, all_builds_file)

        plt.close()

    def run_stepper(self):
        if self.trainer == PpoTrainConfig:
            # PPO
            if self.phase == "P1":
                self.models = {
                    "a": torch.load(
                        "experiments/" + self.mapped_agents.get("a") + "/models/a.pt"
                    ),
                    "p": EmptyModel(
                        self.env.observation_space_pl,
                        7,
                    ),
                }
            else:
                self.models = {
                    "a": torch.load(
                        "experiments/" + self.mapped_agents["a"] + "/models/a.pt"
                    ),
                    "p": torch.load(
                        "experiments/" + self.mapped_agents["p"] + "/models/p.pt"
                    ),
                }

            def stepper():
                def get_actions(obs: dict):
                    actions = {}
                    for key in obs.keys():
                        actions[key], _ = self.models[self.mapping_function(key)].act(
                            obs[key]
                        )

                    return actions

                env = self.env
                env.seed = self.seed
                obs = env.reset(force_dense_logging=True)

                for _ in tqdm(range(env.env.episode_length)):
                    actions = get_actions(obs)
                    obs, rew, done, _ = env.step(actions)

                    if done["__all__"] is True:
                        break

                    with open(self.path + "/logs/Simulation.csv", "a") as log_file:
                        log_file.write(
                            f"{rew['0']},{rew['1']},{rew['2']},{rew['3']},{rew['p']}\n"
                        )

                dense_log = env.env.previous_episode_dense_log

                with open(self.path + "/logs/dense_logs.pkl", "wb") as dense_logs:
                    pickle.dump(dense_log, dense_logs)

                return dense_log

        elif self.trainer == DtTrainConfig:
            if self.phase == "P1":
                self.models = {
                    "a": os.path.join(
                        "experiments", self.mapped_agents.get("a"), "models", "dt_a.pkl"
                    ),
                    "p": None,
                }
            else:
                self.models = {
                    "a": os.path.join(
                        "experiments",
                        self.mapped_agents.get("a"),
                    ),
                    "p": os.path.join(
                        "experiments",
                        self.mapped_agents.get("p"),
                    ),
                }

            def stepper():
                env = self.env
                env.seed = self.seed

                # Done only for intellisense and to remember types
                self.trainer = DtTrainConfig()

                rewards, dense_log = self.trainer.stepper(
                    agent_path=self.models["a"], planner_path=self.models["p"], env=env
                )

                with open(os.path.join(self.path, "logs", "1.csv"), "a") as reward_file:
                    for rew in rewards:
                        reward_file.write(
                            f"{rew['0']},{rew['1']},{rew['2']},{rew['3']},{rew['p']}\n"
                        )

                with open(
                    os.path.join(self.path, "logs", "dense_logs.pkl"), "wb"
                ) as log_file:
                    pickle.dump(dense_log, log_file)

                return dense_log

        elif self.trainer == PPODtTrainConfig:
            if self.phase == "P1":
                self.models = {
                    "a": os.path.join(
                        "experiments", self.mapped_agents.get("a"), "models", "a.pkl"
                    ),
                    "p": None,
                }
            else:
                self.models = {
                    "a": os.path.join(
                        "experiments",
                        self.mapped_agents.get("a"),
                    ),
                    "p": os.path.join(
                        "experiments",
                        self.mapped_agents.get("p"),
                    ),
                }

            def stepper():
                env = self.env
                env.seed = self.seed
                torch.manual_seed(self.seed)
                random.seed(self.seed)
                np.random.seed(self.seed)

                # Done only for intellisense and to remember types
                self.trainer = PPODtTrainConfig()

                rewards, dense_log = self.trainer.stepper(
                    agent_path=self.models["a"], planner_path=self.models["p"], env=env
                )

                with open(
                    os.path.join(self.path, "logs", "dt.csv"), "a"
                ) as reward_file:
                    for rew in rewards:
                        reward_file.write(
                            f"{rew['0']},{rew['1']},{rew['2']},{rew['3']},{rew['p']}\n"
                        )

                with open(
                    os.path.join(self.path, "logs", "dense_logs.pkl"), "wb"
                ) as log_file:
                    pickle.dump(dense_log, log_file)

                return dense_log

        return stepper()

    def setup_logs_and_dirs(self):
        """
        Creates this experiment directory and a log file.
        """

        folder_dir = self.mapped_agents.get("a").split("_")
        date = folder_dir[-3] #+ "_" + folder_dir[-2]
        id = folder_dir[-2]
        algorithm_name = "PPO" 
        if self.trainer ==  DtTrainConfig:
            algorithm_name = "DT"
        if self.trainer ==  PPODtTrainConfig:
            algorithm_name = "PPO_DT"

        experiment_name = (
            f"INT_{algorithm_name}_{date}_{id}"
        )
        self.path = f"experiments/{experiment_name}"

        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(self.path + "/logs")
            os.makedirs(self.path + "/plots")

        # Copy all the self.mapped_agents.get("a") content in the new dir
        shutil.copytree(
            os.path.join("experiments", self.mapped_agents.get("a")),
            self.path,
            dirs_exist_ok=True,
        )

        # # Create config.txt log file
        # with open(self.path + "/config.toml", "w") as config_file:
        #     config_dict = {
        #         "common": {
        #             "algorithm_name": algorithm_name,
        #             "step": self.env.env.episode_length,
        #             "seed": self.seed,
        #             "device": self.device,
        #             "mapped_agents": self.mapped_agents,
        #         }
        #     }

        #     # Algorithm's specific infos
        #     if algorithm_name == {"PPO"}:
        #         # TODO: save PPO's config
        #         # (learning rate but it's kinda useless)

        #         pass
        #     else:
        #         # TODO: save DT's config
        #         pass

        #     toml.dump(config_dict, config_file)

        with open(self.path + "/logs/Simulation.csv", "a+") as log_file:
            log_file.write("0,1,2,3,p\n")

        logging.info("Directories created")
        return experiment_name

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
        self.mapping_function = self.mapping_function()

        if not isinstance(self.seed, int):
            raise ValueError("'seed' muse be integer!")

        if self.device != "cpu":
            # raise ValueError()
            logging.warning(
                "The only available 'device' at the moment is 'cpu'. Redirecting everything to 'cpu'!"
            )
            self.device = "cpu"

        if not self.trainer in [PpoTrainConfig, DtTrainConfig, PPODtTrainConfig]:
            raise ValueError(
                "`self.trainer` must be `PpoTrainConfig` or `DtTrainConfig`!"
            )

        # Check PPO's config
        if self.trainer == PpoTrainConfig:
            # TODO config validation
            pass
        else:
            # Check DT's config
            # TODO: controllo da fare con Andrea.
            pass

        if not isinstance(self.mapped_agents["p"], bool):
            self.phase = "P2"
            if (
                not isinstance(self.mapped_agents["a"], bool)
                and self.mapped_agents["p"] != self.mapped_agents["a"]
            ):
                raise ValueError("The path for pretrained models must be the same!")
        else:
            self.phase = "P1"

        logging.info("Config validation: OK!")
