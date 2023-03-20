"""
Defines the configuration for training with online learning.

Supports DT_ql.
"""
import os
import string
import random
import numpy as np


from datetime import datetime

from src.common import get_environment
from typing import Dict, List, Tuple, Union
from trainer.policies.decision_tree import PythonDT
from ai_economist.foundation.base.base_env import BaseEnvironment
from trainer.policies.grammatical_evolution import (
    GrammaticalEvolutionTranslator,
    grammatical_evolution,
)


class DtTrainConfig:
    """
    Endpoint to setup DT_ql's training configuration.

    Args:
        env: BaseEnvironment
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
        env: BaseEnvironment,
        seed: int = 1,
        lr: Union[float, str] = "auto",
        df: float = 0.9,
        eps: float = 0.05,
        low: float = -10,
        up: float = 10,
        input_space: int = 1,
        episodes: int = 2,  # 500,
        episode_len: int = 3,  # 1000,
        lambda_: int = 2,  # 1000,
        generations: int = 2,  # 1000,
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
    ):

        # Set seeds
        assert seed >= 1, "Seed must be greater than 0"
        np.random.seed(seed)
        random.seed(seed)

        # Add important variables to self
        self.episodes = episodes
        self.episode_len = episode_len
        # For the leaves
        self.n_actions = { 
            'a': env.action_space.n,
            'p': env.action_space_pl.nvec[0].item(),
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
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logdir = "experiments/dt/{}_{}".format(date, "".join(np.random.choice(list(string.ascii_lowercase), size=8)))
        self.logfile = os.path.join(self.logdir, "log.txt")
        os.makedirs(self.logdir)

        # Convert the string to dict
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection

        # Initialize some variables
        grammar = {
            "bt": ["<if>"],
            "if": ["if <condition>:{<action>}else:{<action>}"],
            "condition": [
                "_in_{0}<comp_op><const_type_{0}>".format(k) for k in range(input_space)
            ],
            "action": ['out=_leaf;leaf="_leaf"', "<if>"],
            "comp_op": [" < ", " > "],
        }
        types: str = (
            types
            if types is not None
            else ";".join(["0,10,1,10" for _ in range(input_space)])
        )
        types: str = types.replace("#", "")
        assert (
            len(types.split(";")) == input_space
        ), "Expected {} types, got {}.".format(input_space, len(types.split(";")))

        for index, type_ in enumerate(types.split(";")):
            rng = type_.split(",")
            start, stop, step, divisor = map(int, rng)
            consts_ = list(
                map(str, [float(c) / divisor for c in range(start, stop, step)])
            )
            grammar["const_type_{}".format(index)] = consts_

        # Add to self
        self.grammar = grammar

        # Log the configuration
        with open(self.logfile, "a") as f:
            vars_ = locals().copy()
            for k, v in vars_.items():
                f.write("{}: {}\n".format(k, v))

        
    def __check_variables(self):
        """
        Checks if all variables are set.
        """
        assert not isinstance(self.env, BaseEnvironment), "{} is not known".format(type(self.env))
        assert self.lr == 'auto' or isinstance(self.lr, float), "{} is not known".format(type(self.lr))
        assert self.df > 0 and self.df < 1, "df must be between 0 and 1"
        assert self.eps > 0 and self.eps < 1, "eps must be between 0 and 1"
        assert self.low < self.up, "low must be smaller than up"
        assert self.episodes > 0, "episodes must be greater than 0"
        assert self.episode_len > 0, "episode_len must be greater than 0"
        assert self.lambda_ > 0, "lambda must be greater than 0"
        assert self.generations > 0, "generations must be greater than 0"
        assert self.cxp > 0 and self.cxp < 1, "Crossover probability must be between 0 and 1"
        assert self.mp > 0 and self.mp < 1, "Mutation probability must be between 0 and 1"
        assert self.genotype_len > 0, "Genotype length must be greater than 0"


    def evaluate_fitness(
        self,
        fitness_function: callable,
        genotype: List[int],
    ) -> float:
        # Get the phenotype
        phenotype, _ = GrammaticalEvolutionTranslator(self.grammar).genotype_to_str(
            genotype
        )

        # Get the Decision Trees of agents and planner
        dt = PythonDT(
            phenotype,
            self.n_actions.get("a", 50),
            self.lr,
            self.df,
            self.eps,
            self.low,
            self.up,
        )
        dt_p = PythonDT(
            phenotype,
            self.n_actions.get("p", 22),
            self.lr,
            self.df,
            self.eps,
            self.low,
            self.up,
            True,
        )

        # Evaluate the fitness
        return fitness_function(dt, dt_p)

    def fitness(self, agent: PythonDT, planner: PythonDT):
        global_cumulative_rewards = []
        e = get_environment()

        # try:
        for iteration in range(self.episodes):
            # Set the seed and reset the environment
            e.seed(iteration)
            obs: Dict[str, Dict[str, np.ndarray]] = e.reset()

            # Start the episode
            agent.new_episode()
            planner.new_episode()

            # Initialize some variables
            cum_rew = 0

            # Run the episode
            for t in range(self.episode_len):
                actions = {}
                for k_agent, _obs in obs.items():
                    if k_agent == "p":
                        actions[k_agent] = planner(_obs.get("flat"))
                        if actions[k_agent] is None:
                            actions[k_agent] = [0 for _ in range(7)]
                    else:
                        actions[k_agent] = agent(_obs.get("flat"))
                        if actions[k_agent] is None:
                            actions[k_agent] = 0

                if any([a is None for a in actions.values()]):
                    break

                obs, rew, done, _ = e.step(actions)

                # e.render() # FIXME: This is not working, see if needed
                agent.set_reward(sum([rew.get(k, 0) for k in rew.keys() if k != "p"]))
                planner.set_reward(rew.get("p", 0))

                # cum_rew += rew
                cum_rew += sum([rew.get(k, 0) for k in rew.keys()])

                if done:
                    break

            # FIXME: I added this but I think it is wrong
            # agent.set_reward(sum([rew.get(k, 0) for k in rew.keys() if k != "p"]))
            # planner.set_reward(rew.get("p", 0))

            for k_agent, _obs in obs.items():
                if k_agent == "p":
                    planner(_obs.get("flat"))
                else:
                    agent(_obs.get("flat"))
            global_cumulative_rewards.append(cum_rew)
        # except Exception as ex:
        #     raise ex
        #     if len(global_cumulative_rewards) == 0:
        #         global_cumulative_rewards = -1000
        # e.close()

        fitness = (np.mean(global_cumulative_rewards),)
        return fitness, agent.leaves, planner.leaves

    def train(
        self,
    ) -> None:
        def fit_fcn(x: List[int]):
            return self.evaluate_fitness(self.fitness, x)

        # with parallel_backend("multiprocessing"):
        pop, log, hof, best_leaves = grammatical_evolution(
            fit_fcn,
            individuals=self.lambda_,
            generations=self.generations,
            jobs=1,  # TODO: This should be removed because >1 is not working
            cx_prob=self.cxp,
            m_prob=self.mp,
            logfile=self.logfile,
            # seed=self.seed,  # This is useless since the seed is already set in the __init__
            mutation=self.mutation,
            crossover=self.crossover,
            initial_len=self.genotype_len,
            selection=self.selection,
        )

        # Log best individual
        with open(self.logfile, "a") as log_:
            phenotype, _ = GrammaticalEvolutionTranslator(self.grammar).genotype_to_str(
                hof[0]
            )
            phenotype = phenotype.replace('leaf="_leaf"', "")

            for k in range(50000):  # Iterate over all possible leaves
                key = "leaf_{}".format(k)
                if key in best_leaves:
                    v = best_leaves[key].q
                    phenotype = phenotype.replace(
                        "out=_leaf", "out={}".format(np.argmax(v)), 1
                    )
                else:
                    break

            log_.write(str(log) + "\n")
            log_.write(str(hof[0]) + "\n")
            log_.write(phenotype + "\n")
            log_.write("best_fitness: {}".format(hof[0].fitness.values[0]))

        with open(os.path.join(self.logdir, "fitness.tsv"), "w") as f:
            f.write(str(log))
