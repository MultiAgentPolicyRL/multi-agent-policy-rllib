"""
TODO
"""
from datetime import datetime
import os
import pickle
import time
import toml

from tqdm import tqdm
from src.common import test_mapping, EmptyModel
from src.train.dt_ql_train import DtTrainConfig
from src.train.ppo_train import PpoTrainConfig
import logging
import torch


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
        self.mapping_function = mapping_function  #
        self.env = env  #
        self.device = device  #
        self.trainer = trainer  #
        self.seed = seed  #
        self.config = config
        self.mapped_agents = mapped_agents  #

        self.validate_config(env=env)
        self.setup_logs_and_dirs()
        self.build_stepper()

        # Build correct stepper (build and load specific models)
        stepper = self.build_stepper()
        # Do stepping and save the log
        stepper()

    def build_stepper(self):
        if self.trainer == PpoTrainConfig:
            ### PPO
            # Load models. If P1: load `a`, if P2: load `a`,`p`
            if self.phase == "P1":
                self.models = {
                    "a": torch.load(
                        "/experiments/" + self.mapped_agents("a") + "/models/a.pt"
                    ),
                    "p": EmptyModel(
                        self.env.observation_space_pl,
                        [22, 22, 22, 22, 22, 22, 22],
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

                    with open(self.path + "/logs/-1.csv", "a") as log_file:
                        log_file.write(
                            f"{rew['0']},{rew['1']},{rew['2']},{rew['3']},{rew['p']}\n"
                        )

                dense_log = env.env.previous_episode_dense_log

                with open(self.path + "/logs/dense_logs", "wb") as dense_logs:
                    pickle.dump(dense_log, dense_logs)

            return stepper

        else:
            ### DT
            pass

    def setup_logs_and_dirs(self):
        """
        Creates this experiment directory and a log file.
        """

        date = datetime.today().strftime("%d-%m-%Y")
        algorithm_name = "PPO" if self.trainer == PpoTrainConfig else "DT"
        experiment_name = (
            f"INT_{algorithm_name}_{date}_{int(time.time())}_{self.mapped_agents['a']}"
        )
        self.path = f"experiments/{experiment_name}"

        if not os.path.exists(self.path):
            os.makedirs(self.path)
            os.makedirs(self.path + "/logs")
            os.makedirs(self.path + "/plots")

        # Create config.txt log file
        with open(self.path + "/config.toml", "w") as config_file:
            config_dict = {
                "common": {
                    "algorithm_name": algorithm_name,
                    "step": self.env.env.episode_length,
                    "seed": self.seed,
                    "device": self.device,
                    "mapped_agents": self.mapped_agents,
                }
            }

            # Algorithm's specific infos
            if algorithm_name == {"PPO"}:
                # TODO: save PPO's config
                # (learning rate but it's kinda useless)

                pass
            else:
                # TODO: save DT's config
                pass

            toml.dump(config_dict, config_file)

        with open(self.path + "/logs/-1.csv", "a+") as log_file:
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

        if not self.trainer in [PpoTrainConfig, DtTrainConfig]:
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
            if self.mapped_agents["p"] != self.mapped_agents["a"]:
                raise ValueError("The path for pretrained models must be the same!")
        else:
            self.phase = "P1"

        logging.info("Config validation: OK!")
