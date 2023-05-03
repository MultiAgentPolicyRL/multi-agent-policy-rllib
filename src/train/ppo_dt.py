"""
Defines the configuration for training with online learning.

Supports DT_ql.
"""
import logging
import os
import toml
import torch
import random
import numpy as np


from datetime import datetime

from tqdm import tqdm
from joblib import parallel_backend

from typing import Dict, List, Tuple, Union
from src.train.dt import PythonDT
from src.common.env import EnvWrapper
from ai_economist.foundation.base.base_env import BaseEnvironment
from src.common.rollout_buffer import RolloutBuffer
from src.train.ppo import PpoPolicy

from src.train.dt import (
    GrammaticalEvolutionTranslator,
    grammatical_evolution,
)
from src.train.ppo.models.linear import PytorchLinearA


class PPODtTrainConfig:
    """
    Endpoint to setup DT_ql's training configuration.

    Args:
        env: EnvWrapper
            The environment to train on. Should be the AI-Economist environment.
        seed: int
            The seed to use for reproducibility.
        lr: Union[float, str]
            The learning rate to use. If "auto", the learning rate will be automatically determined.
        df: float
            The discount factor to use.
        eps: float
            The epsilon value to use for epsilon-greedy.
        low: float
            The lower bound for the action space.
        up: float
            The upper bound for the action space.
        input_space: int
            The number of inputs to the decision tree.
        episodes: int
            The number of episodes to train for.
        episode_len: int
            The number of steps per episode.
        lambda_: int
            The lambda value to use for the evolution.
        generations: int
            The number of generations to train for.
        cxp: float
            The crossover probability to use.
        mp: float
            The mutation probability to use.
        mutation: Dict[str, Union[str, int, float]]
            The mutation function to use.
        crossover: Dict[str, Union[str, int, float]]
            The crossover function to use.
        selection: Dict[str, Union[str, int, float]]
            The selection function to use.
        genotype_len: int
            The length of the genotype.
        types: List[Tuple[int, int, int, int]]
            The types to use for the decision tree.
    """

    def __init__(
        self,
        env: EnvWrapper = None,
        agent: Union[bool, str] = True,
        planner: Union[bool, str] = True,
        seed: int = 1,
        lr: Union[float, str] = "auto",
        df: float = 0.9,
        eps: float = 0.05,
        low: float = -10,
        up: float = 10,
        input_space: int = 1,
        episodes: int = 500,
        episode_len: int = 1000,
        lambda_: int = 1000,
        generations: int = 1000,
        cxp: float = 0.5,
        mp: float = 0.5,
        mutation: Dict[str, Union[str, int, float]] = {
            "function": "tools.mutUniformInt",
            "low": 0,
            "up": 40000,
            "indpb": 0.1,
        },
        crossover: Dict[str, Union[str, int, float]] = {"function": "tools.cxOnePoint"},
        selection: Dict[str, Union[str, int, float]] = {
            "function": "tools.selTournament",
            "tournsize": 2,
        },
        genotype_len: int = 100,
        types: List[Tuple[int, int, int, int]] = None,
        batch_size: int = 200,
        rollout_fragment_length: int = 200,
        k_epochs: int = 16,
        eps_clip: int = 10,
        gamma: float = 0.998,
        device: str = "cpu",
        learning_rate: float = 0.0003,
        _c1: float = 0.05,
        _c2: float = 0.025,
        mapped_agents: Dict[str, Union[bool, str]] = {
            "a": True,
            "p": True,
        },
    ):
        

        if env is not None:
            # PPO specific stuff
            self.rollout_fragment_length = rollout_fragment_length
            self.batch_size = batch_size
            self.step = episode_len
            self.batch_iterations = self.batch_size//self.rollout_fragment_length
            self.k_epochs = k_epochs
            self.eps_clip = eps_clip
            self.gamma = gamma
            self.device = device
            self.learning_rate = learning_rate
            self._c1 = _c1
            self._c2 = _c2
            
            self.agent = PpoPolicy(env.observation_space, 50)
            self.agent.load_model("experiments/"+mapped_agents["a"]+"/models/a.pt")
            
            obs = env.reset()
            self.memory = RolloutBuffer(obs, None)
            self.rolling_memory = RolloutBuffer(obs, None)
            del obs

            # DT specific stuff
            self.env = env
            # self.agent = mapped_agents["a"]
            self.planner = mapped_agents["p"]

            # Set seeds
            assert seed >= 1, "Seed must be greater than 0"
            np.random.seed(seed)
            random.seed(seed)
            self.env.seed = seed
            self.seed = seed

            # Add important variables to self
            self.episodes = episodes
            self.episode_len = episode_len
            # For the leaves
            self.n_actions = {
                "a": env.action_space.n,
                "p": env.action_space_pl.nvec[0].item(),
            }
            # For the input space
            obs = env.reset()
            input_space = {
                "a": obs.get('0').get('flat').shape[0],
                "p": obs.get('p').get('flat').shape[0]
            }
            self.lr = lr
            self.df = df
            self.eps = eps
            self.low = low
            self.up = up

            # For the evolution
            self.lambda_ = lambda_
            self.generations = generations
            self.cxp = cxp
            self.mp = mp
            self.genotype_len = genotype_len
            self.types = types

            # Create the log directory
            phase = "P1" if agent and not planner else "P2"
            date = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.logdir = f"experiments/PPO_DT_{phase}_{date}_{episodes}"
            self.logfile = os.path.join(self.logdir, "log.txt")
            os.makedirs(self.logdir)

            # Convert the string to dict
            self.mutation = mutation
            self.crossover = crossover
            self.selection = selection

            # Initialize some variables
            grammar_agent = {
                "bt": ["<if>"],
                "if": ["if <condition>:{<action>}else:{<action>}"],
                "condition": [
                    "_in_{0}<comp_op><const_type_{0}>".format(k) for k in range(input_space.get("a", 136))
                ],
                "action": ['out=_leaf;leaf="_leaf"', "<if>"],
                "comp_op": [" < ", " > "],
            }
            types_agent: str = (
                types
                if types is not None
                else ";".join(["0,10,1,10" for _ in range(input_space.get("a", 136))])
            )
            types_agent: str = types_agent.replace("#", "")
            assert (
                len(types_agent.split(";")) == input_space.get("a", 136)
            ), "Expected {} types_agent, got {}.".format(input_space.get("a", 136), len(types_agent.split(";")))

            for index, type_ in enumerate(types_agent.split(";")):
                rng = type_.split(",")
                start, stop, step, divisor = map(int, rng)
                consts_ = list(
                    map(str, [float(c) / divisor for c in range(start, stop, step)])
                )
                grammar_agent["const_type_{}".format(index)] = consts_

            # Add to self
            self.grammar_agent = grammar_agent

            grammar_planner = {
                "bt": ["<if>"],
                "if": ["if <condition>:{<action>}else:{<action>}"],
                "condition": [
                    "_in_{0}<comp_op><const_type_{0}>".format(k) for k in range(input_space.get("p", 86))
                ],
                "action": ['out=_leaf;leaf="_leaf"', "<if>"],
                "comp_op": [" < ", " > "],
            }
            types_planner: str = (
                types
                if types is not None
                else ";".join(["0,10,1,10" for _ in range(input_space.get("p", 86))])
            )
            types_planner: str = types_planner.replace("#", "")
            assert (
                len(types_planner.split(";")) == input_space.get("p", 86)
            ), "Expected {} types_planner, got {}.".format(input_space.get("p", 86), len(types_planner.split(";")))

            for index, type_ in enumerate(types_planner.split(";")):
                rng = type_.split(",")
                start, stop, step, divisor = map(int, rng)
                consts_ = list(
                    map(str, [float(c) / divisor for c in range(start, stop, step)])
                )
                grammar_planner["const_type_{}".format(index)] = consts_

            # Add to self
            self.grammar_planner = grammar_planner

            # Log the configuration
            with open(os.path.join(self.logdir, "config.toml"), "w") as f:
                config_dict = {
                    "common": {
                        "algorithm_name": "PPO_DT",
                        "phase": 1 if not planner else 2,
                        "step": self.episode_len,
                        "seed": seed,
                        "device": "cpu",
                        "mapped_agents": mapped_agents,
                    },
                    "algorithm_specific": {
                        "dt": {
                        "episodes": episodes,
                        "generations": generations,
                        "cxp": cxp,
                        "mp": mp,
                        "genotype_len": genotype_len,
                        "types": types,
                        "mutation": self.mutation,
                        "crossover": self.crossover,
                        "selection": self.selection,
                        "lr": lr,
                        "df": df,
                        "eps": eps,
                        "low": low,
                        "up": up,
                        "lambda_": lambda_,
                        "input_space": input_space,
                        "grammar_agent": self.grammar_agent,
                        "grammar_planner": self.grammar_planner,
                        },
                        "ppo": {
                            "rollout_fragment_length": self.rollout_fragment_length,
                            "batch_size": self.batch_size,
                            "k_epochs": self.k_epochs,
                            "eps_clip": self.eps_clip,
                            "gamma": self.gamma,
                            "learning_rate": self.learning_rate,
                            "c1": self._c1,
                            "c2": self._c2,
                        }
                    },
                }
                toml.dump(config_dict, f)

            # Check the variables
            self.__check_variables()
            self.validate_config(env=env)

    def validate_config(self, env):
        """
        Validate PPO's config.

        Raise:
            ValueError if a specific parameter is not set correctly.
        """
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

        logging.info("Configuration validated: OK")
        return

    def __check_variables(self):
        """
        Checks if all variables are set.
        """
        assert self.lr == "auto" or (
            isinstance(self.lr, float) and self.lr > 0 and self.lr < 1
        ), "{} is not known or not in the right range ({})".format(
            type(self.lr), self.lr
        )
        assert self.df > 0 and self.df < 1, "df must be between 0 and 1"
        assert self.eps > 0 and self.eps < 1, "eps must be between 0 and 1"
        assert self.low < self.up, "low must be smaller than up"
        assert self.episodes > 0, "episodes must be greater than 0"
        assert self.episode_len > 0, "episode_len must be greater than 0"
        assert self.lambda_ > 0, "lambda must be greater than 0"
        assert self.generations > 0, "generations must be greater than 0"
        assert (
            self.cxp > 0 and self.cxp < 1
        ), "Crossover probability must be between 0 and 1"
        assert (
            self.mp > 0 and self.mp < 1
        ), "Mutation probability must be between 0 and 1"
        assert self.genotype_len > 0, "Genotype length must be greater than 0"

        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir, exist_ok=True)

        models_dir = os.path.join(self.logdir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)

        rewards_dir = os.path.join(self.logdir, "rewards")
        if not os.path.exists(rewards_dir):
            os.makedirs(rewards_dir, exist_ok=True)

        rewards_csv_file = os.path.join(rewards_dir, "dt.csv")
        with open(rewards_csv_file, "w") as f:
            f.write("0,1,2,3,p\n")

        with open(f"{self.logdir}/rewards/ppo.csv", "w") as f:
            f.write("0,1,2,3,p\n")

    def evaluate_fitness(
        self,
        genotype: List[int],
    ) -> float:
        # Get the phenotype
        phenotype_planner, _ = GrammaticalEvolutionTranslator(self.grammar_planner).genotype_to_str(
            genotype
        )

        dt_p = PythonDT(
            phenotype_planner,
            self.n_actions.get("p", 22),
            self.lr,
            self.df,
            self.eps,
            self.low,
            self.up,
            planner=True,
        )

        # Evaluate the fitness
        return self.fitness(dt_p)

    def batch(self, planner):
        """
        Creates a batch of `rollout_fragment_length` steps, 
        """
        # reset batching environment and get its observation
        obs = self.env.reset()

        # reset rollout_buffer
        self.memory.clear()

        for _ in tqdm(range(self.batch_iterations)):
            for _ in range(self.rollout_fragment_length):
                # get actions, action_logprob for all agents in each policy* wrt observation
                policy_action, policy_logprob = self.get_actions(obs, planner)

                # get new_observation, reward, done from stepping the environment
                next_obs, rew, done, _ = self.env.step(policy_action)

                if done["__all__"] is True:
                    next_obs = self.env.reset()

                policy_action["p"]=np.zeros((1))

                self.rolling_memory.update(
                    action=policy_action,
                    logprob=policy_logprob,
                    state=obs,
                    reward=rew,
                    is_terminal=done["__all__"],
                )

                obs = next_obs
            
            self.memory.extend(self.rolling_memory)
            self.rolling_memory.clear()

            # log sum of reward per agent
            # data = f"{rew['0'].item()},{rew['1'].item()},{rew['2'].item()},{rew['3'].item()},{rew['p'].item()}\n"
    
            # with open(f"{self.logdir}/rewards/ppo.csv", "a+") as file:
                    # file.write(data)
            data = []
            data[0] = (self.memory.buffers["0"].rewards.sum()).item()
            data[1] = (self.memory.buffers["1"].rewards.sum()).item()
            data[2] = (self.memory.buffers["2"].rewards.sum()).item()
            data[3] = (self.memory.buffers["3"].rewards.sum()).item()
            data[4] = (self.memory.buffers["p"].rewards.sum()).item()
            
            data1 = f"{data[0]},{data[1]},{data[2]},{data[3]},{data[4]}\n"
            with open(f"{self.logdir}/rewards/ppo.csv", "a+") as file:
                file.write(data1)

    def __append_tensor(self, old_stack, new_tensor):
        """
        Appends two tensors on a new axis. If `old_stack` shape is the same of `new_tensor`
        they are stacked together with `np.stack`. Otherwise if `old_stack` shape is bigger
        than `new_tensor` shape's, `new_tensor` is expanded on axis 0 and they are concatenated.

        Args:
            old_stack: old stack, it's shape must be >= new_tensor's shape
            new_tensor: new tensor that will be appended to `old_stack`
        """
        shape = (1, 1)
        if new_tensor.shape:
            shape = (1, *new_tensor.shape)

        nt = np.reshape(new_tensor, shape)

        if old_stack is None or old_stack is {}:
            return nt
        else:
            return np.concatenate((old_stack, nt))

    def get_actions(self, obs, planner):
        policy_probs = None
        policy_actions = []
        for x in ['0','1','2','3']:
            a, b = self.agent.act(obs[x])
            policy_actions.append(a)
            policy_probs = self.__append_tensor(policy_probs, b)
        
        actions = {
            '0': policy_actions[0],
            '1': policy_actions[1],
            '2': policy_actions[2],
            '3': policy_actions[3],
            'p': planner
        }

        probs = {
            '0': policy_probs[0],
            '1': policy_probs[1],
            '2': policy_probs[2],
            '3': policy_probs[3],
            'p': np.zeros((1))
        }

        return actions, probs


    def get_actions_ppo_only(self, obs):
        policy_actions = []
        for x in ['0','1','2','3']:
            a, _ = self.agent.act(obs[x])
            policy_actions.append(a)

        actions = {
            '0': policy_actions[0],
            '1': policy_actions[1],
            '2': policy_actions[2],
            '3': policy_actions[3],
        }

        return actions

    def stepper(self, agent_path: str, planner_path: str = None, env: EnvWrapper = None): 
        """
        Stepper used for the `interact.py`. N.B.: In both agent and planner paths the prefix `experiments/` is already added.

        ---

        Parameters:

        agent_path: str
            Path to the agent's folder.
        planner_path: str
            Path to the planner's folder.
        env: BaseEnvironment
            Environment used for the interaction.

        ---

        Returns:

        Tuple[List[float], Dict[str, np.ndarray]]
            Tuple containing the list of the rewards and the dictionary containing the
            observations.

        ---

        """

        _agent_path = os.path.join(agent_path, "models", "a.pt")
        if not os.path.exists(_agent_path):
            _agent_path = os.path.join("experiments", "PPO_P1_01-04-2023_1680328509_1000", "models", "a.pt")
            if not os.path.exists(_agent_path):
                raise FileNotFoundError(f"No agent found neither in '{agent_path}' nor in 'PPO_P1_01-04-2023_1680328509_1000'.")
            
        agent: PytorchLinearA = torch.load(_agent_path)
        planner = PythonDT(load_path=os.path.join(planner_path, "models", "dt_p.pkl"), planner=True) 

        # Initialize some variables
        obs: Dict[str, Dict[str, np.ndarray]] = env.reset(force_dense_logging=True)

        # Run the episode
        for t in tqdm(range(env.env.episode_length), desc="DecisionTree Replay"):
            with torch.no_grad():
                actions = agent.get_actions(obs)
            actions["p"] = np.zeros((7))

            obs, rew, done, _ = env.step(actions)
            planner.add_rewards(rewards=rew)

            if done["__all__"] is True:
                break

        if not done["__all__"]:
            # Start the episode
            planner.new_episode()

            planner_action = planner(obs.get("p").get("flat"))
            actions = {
                key: [0] for key in obs.keys()
            }
            actions['p'] = planner_action
            obs, rew, done, _ = env.step(actions)
            
            planner.set_reward(rew.get("p", 0))

        return planner.rewards, env.env.previous_episode_dense_log

    def fitness(self, planner: PythonDT):
        global_cumulative_rewards = []

        for _ in range(self.episodes):
            # create batch for PPO learning

            obs: Dict[str, Dict[str, np.ndarray]] = self.env.reset()
            pa = planner(obs.get("p").get("flat"))
            self.batch(pa)

            # learn on DT
            # Set the seed and reset the environment
            self.seed+=1
            self.env.seed = self.seed

            # Initialize some variables
            cum_global_rew = 0
            cum_rew = {
                key: np.empty((0)) for key in obs.keys()
            }
            # static - one per epoch - action by the planner
            
            planner.new_episode()
            # Run the episode

            for t in range(self.episode_len):
                # Start the episode
                actions = self.get_actions_ppo_only(obs)
                actions['p'] = planner(obs.get("p").get("flat"))
                obs, rew, done, _ = self.env.step(actions)

                for key in obs.keys():
                    cum_rew[key] = np.concatenate((cum_rew[key], rew.get(key, np.nan)))

                cum_global_rew += np.sum([rew.get(k, 0) for k in rew.keys()])
                
                if done["__all__"].item() is True:
                    obs = self.env.reset()
        
                planner.set_reward(rew.get("p", 0))
                global_cumulative_rewards.append(cum_global_rew)

            new_rewards = {}
            for key in obs.keys():
                new_rewards[key] = np.sum(cum_rew.get(key, np.nan))

            planner.add_rewards(rewards=new_rewards)

        fitness = (np.mean(global_cumulative_rewards),)
        
        # learn on PPO
        self.agent.learn(self.memory.to_tensor()["a"])
        return fitness, planner

    def train(
        self,
    ) -> None:
        with parallel_backend("multiprocessing"):
            pop, log, hof, best_leaves = grammatical_evolution(
                self.evaluate_fitness,
                individuals=self.lambda_,
                generations=self.generations,
                cx_prob=self.cxp,
                m_prob=self.mp,
                logfile=self.logfile,
                mutation=self.mutation,
                crossover=self.crossover,
                initial_len=self.genotype_len,
                selection=self.selection,
                planner_only=True,
            )

