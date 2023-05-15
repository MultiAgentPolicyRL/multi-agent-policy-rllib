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

        return        

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

        folder_dir = self.mapped_agents.get("p").split("_")
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
            os.path.join("experiments", self.mapped_agents.get("p")),
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
            # if (
            #     not isinstance(self.mapped_agents["a"], bool)
            #     and self.mapped_agents["p"] != self.mapped_agents["a"]
            # ):
            #     raise ValueError("The path for pretrained models must be the same!")
        else:
            self.phase = "P1"

        logging.info("Config validation: OK!")

    def output_plots(self, dense_logs: dict) -> None:
        """
        Plot the rewards and save the plots in the experiment's directory.
        """

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

        total_tax_paid = 0
        total_income = 0
        with open(
            os.path.join(self.path, "logs", "PeriodicTax.log"), "+w"
        ) as f:
            for idx, tax in enumerate(dense_logs.get('PeriodicTax')):
                if len(tax) > 0:
                    f.write(f"Step {idx+1} has (income, tax_paid): \n\t")
                    for key, values in tax.items():
                        if key in ['0', '1', '2', '3']:
                            f.write(f"{key}: ({values.get('income', -1):.2f}, {values.get('tax_paid', -1):.2f}), ")
                            total_tax_paid += values.get('tax_paid', 0)
                            total_income += values.get('income', 0)
                    f.write("\n\n")

            f.write(f"\nTotal paid taxes are {total_tax_paid:.2f} over the total income {total_income:.2f} with ratio ({total_tax_paid/total_income:.2f})")

        agent_1 = np.empty((0))
        agent_2 = np.empty((0))
        agent_3 = np.empty((0))
        agent_4 = np.empty((0))
        planner = np.empty((0))

        for val in dense_logs.get('rewards'):
            agent_1 = np.append(agent_1, val.get('0', -np.inf))
            agent_2 = np.append(agent_2, val.get('1', -np.inf))
            agent_3 = np.append(agent_3, val.get('2', -np.inf))
            agent_4 = np.append(agent_4, val.get('3', -np.inf))
            planner = np.append(planner, val.get('p', -np.inf))

        all_sum = np.sum([agent_1, agent_2, agent_3, agent_4, planner])

        fig, axs = plt.subplots(3, 2, figsize=(20, 30))

        axs[0][0].scatter(np.arange(1, agent_1.shape[0]+1), agent_1, label='Rewards')
        agent_sum = []
        for rew in agent_1:
            to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
            agent_sum.append(to_add)
        axs[0][0].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
        axs[0][0].set_title('Agent 1')
        axs[0][0].set_xlabel('Episode')
        axs[0][0].set_ylabel('Reward')
        axs[0][0].legend()

        axs[0][1].scatter(np.arange(1, agent_2.shape[0]+1), agent_2, label='Rewards')
        agent_sum = []
        for rew in agent_2:
            to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
            agent_sum.append(to_add)
        axs[0][1].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
        axs[0][1].set_title('Agent 2')
        axs[0][1].set_xlabel('Episode')
        axs[0][1].set_ylabel('Reward')
        axs[0][1].legend()

        axs[1][0].scatter(np.arange(1, agent_3.shape[0]+1), agent_3, label='Rewards')
        agent_sum = []
        for rew in agent_3:
            to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
            agent_sum.append(to_add)
        axs[1][0].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
        axs[1][0].set_title('Agent 3')
        axs[1][0].set_xlabel('Episode')
        axs[1][0].set_ylabel('Reward')
        axs[1][0].legend()

        axs[1][1].scatter(np.arange(1, agent_4.shape[0]+1), agent_4, label='Rewards')
        agent_sum = []
        for rew in agent_4:
            to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
            agent_sum.append(to_add)
        axs[1][1].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
        axs[1][1].set_title('Agent 4')
        axs[1][1].set_xlabel('Episode')
        axs[1][1].set_ylabel('Reward')
        axs[1][1].legend()

        axs[2][0].scatter(np.arange(1, planner.shape[0]+1), planner, label='Rewards')
        agent_sum = []
        for rew in planner:
            to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
            agent_sum.append(to_add)
        axs[2][0].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
        axs[2][0].set_title('Planner')
        axs[2][0].set_xlabel('Episode')
        axs[2][0].set_ylabel('Reward')
        axs[2][0].legend()

        total = []
        for a1, a2, a3, a4, p in zip(agent_1, agent_2, agent_3, agent_4, planner):
            total.append(a1+a2+a3+a4+p)

        axs[2][1].scatter(np.arange(1, len(total)+1), total, label='Rewards')
        agent_sum = []
        for rew in total:
            to_add = np.sum(agent_sum[-1]+rew) if len(agent_sum) > 0 else rew
            agent_sum.append(to_add)
        axs[2][1].plot(np.arange(1, len(agent_sum)+1), agent_sum, label='Rewards Sum')
        axs[2][1].set_title('Total')
        axs[2][1].set_xlabel('Episode')
        axs[2][1].set_ylabel('Reward')
        axs[2][1].legend()


        fig.savefig(os.path.join(self.path, 'plots', 'Rewards.png'))

        plt.close(fig)

        return