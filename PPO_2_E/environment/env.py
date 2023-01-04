"""
Wrapper for making the gather-trade-build environment an OpenAI compatible environment.
"""
import os
import pickle
import random
import sys
from typing import Dict
import warnings

import numpy as np
from ai_economist import foundation
from gym import spaces
from gym.utils import seeding
from environment import env_config
import torch
from tensordict import TensorDict

_BIG_NUMBER = 1e20

# TODO: linting and cleaning


def get_environment(device):
    """
    Returns builded environment with `env_config` config
    """
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    return EnvWrapper(env_config=env_config, device=device)


def recursive_list_to_np_array(d):
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, list):
                new_d[k] = np.array(v)
            elif isinstance(v, dict):
                new_d[k] = recursive_list_to_np_array(v)
            elif isinstance(v, (float, int, np.floating, np.integer)):
                new_d[k] = np.array([v])
            elif isinstance(v, np.ndarray):
                new_d[k] = v
            else:
                raise AssertionError
        return new_d
    raise AssertionError


def pretty_print(dictionary):
    for key in dictionary:
        print("{:15s}: {}".format(key, dictionary[key].shape))
    print("\n")


class EnvWrapper:
    """
    This wrapper adds the action and observation space to the environment,
    and adapts the reset and step functions to run with RLlib.
    """

    def __init__(self, env_config, device, verbose=False):
        self.device = device
        self.env_config_dict = env_config["env_config_dict"]

        # Adding env id in the case of multiple environments
        if hasattr(env_config, "worker_index"):
            self.env_id = (
                env_config["num_envs_per_worker"] * (env_config.worker_index - 1)
            ) + env_config.vector_index
        else:
            ### Modified from None
            self.env_id = 1

        self.env = foundation.make_env_instance(**self.env_config_dict)
        self.verbose = verbose
        self.sample_agent_idx = str(self.env.all_agents[0].idx)

        obs = self.env.reset()

        self.observation_space = self._dict_to_spaces_dict(obs["0"])
        self.observation_space_pl = self._dict_to_spaces_dict(obs["p"])

        if self.env.world.agents[0].multi_action_mode:
            self.action_space = spaces.MultiDiscrete(
                self.env.get_agent(self.sample_agent_idx).action_spaces
            )
            self.action_space.dtype = np.int64
            self.action_space.nvec = self.action_space.nvec.astype(np.int64)

        else:
            self.action_space = spaces.Discrete(
                self.env.get_agent(self.sample_agent_idx).action_spaces
            )
            self.action_space.dtype = np.int64

        if self.env.world.planner.multi_action_mode:
            self.action_space_pl = spaces.MultiDiscrete(
                self.env.get_agent("p").action_spaces
            )
            self.action_space_pl.dtype = np.int64
            self.action_space_pl.nvec = self.action_space_pl.nvec.astype(np.int64)

        else:
            self.action_space_pl = spaces.Discrete(
                self.env.get_agent("p").action_spaces
            )
            self.action_space_pl.dtype = np.int64

        # Created | build global_action_space and global_observation_space
        # self.env_config_dict["n_agents"]
        global_observation_space = {}
        global_action_space = {}
        for n in range(self.env_config_dict["n_agents"]):
            global_observation_space[str(n)] = self.observation_space
            global_action_space[str(n)] = self.action_space
        global_observation_space["p"] = self.observation_space_pl
        global_action_space["p"] = self.action_space_pl

        self.global_observation_space = global_observation_space
        self.global_action_space = global_action_space

        self._seed = None
        if self.verbose:
            print("[EnvWrapper] Spaces")
            print("[EnvWrapper] Obs (a)   ")
            pretty_print(self.observation_space)
            print("[EnvWrapper] Obs (p)   ")
            pretty_print(self.observation_space_pl)
            print("[EnvWrapper] Action (a)", self.action_space)
            print("[EnvWrapper] Action (p)", self.action_space_pl)

    def _dict_to_spaces_dict(self, obs):
        dict_of_spaces = {}
        for k, v in obs.items():

            # list of lists are listified np arrays
            _v = v
            if isinstance(v, list):
                _v = np.array(v)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                _v = np.array([v])

            # assign Space
            if isinstance(_v, np.ndarray):
                x = float(_BIG_NUMBER)
                # Warnings for extreme values
                if np.max(_v) > x:
                    warnings.warn("Input is too large!")
                if np.min(_v) < -x:
                    warnings.warn("Input is too small!")
                box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                # This loop avoids issues with overflow to make sure low/high are good.
                while not low_high_valid:
                    x = x // 2
                    box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                    low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                dict_of_spaces[k] = box

            elif isinstance(_v, dict):
                dict_of_spaces[k] = self._dict_to_spaces_dict(_v)
            else:
                raise TypeError
        return spaces.Dict(dict_of_spaces)

    def data_preprocess(self, observation: dict) -> dict:
        """
        Takes as an input a dict of np.arrays and trasforms them to Torch.tensors.

        Args:
            observation: observation of the environment

        Returns:
            observation_tensored: same structure of `observation`, but np.arrays are not
                torch.tensors

            observation_tensored: {
                '0': {
                    'var': Tensor
                    ...
                },
                ...
            }
        """
        # observation_tensored = observation

        observation_tensored = {}

        for key in observation:
            # Agents: '0', '1', '2', '3', 'p'
            # observation_tensored[key] = {}
            # for data_key in observation[key]:
            #     print(observation[key][data_key].shape)    
            # # # Accessing to specific data like 'world-map', 'flat', 'time', ...
            # #     observation_tensored[key][data_key] = (
            # #         torch.Tensor(observation[key][data_key]).unsqueeze(0).long().to(self.device)
            # #     )
            # sys.exit()
            observation_tensored[key] = TensorDict(observation[key], batch_size=[]).to(self.device)
            # if key != 'p':
            #     observation_tensored[key] = TensorDict(observation[key], batch_size=[]).to(self.device)
            # else:
            #     observation_tensored[key] = TensorDict(observation[key], batch_size=[]).to(self.device)
                # banana:TensorDict=TensorDict(observation[key], batch_size=[]).to(self.device)
            # print(observation_tensored[key].shape)
            # banana.shape
        # observation_tensored = TensorDict(observation, batch_size=[1, 2, 136, 7, 50])
        return observation_tensored

    @property
    def pickle_file(self):
        if self.env_id is None:
            return "game_object.pkl"
        return "game_object_{:03d}.pkl".format(self.env_id)

    def save_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        with open(path, "wb") as F:
            pickle.dump(self.env, F)

    def load_game_object(self, save_dir):
        assert os.path.isdir(save_dir)
        path = os.path.join(save_dir, self.pickle_file)
        with open(path, "rb") as F:
            self.env = pickle.load(F)

    @property
    def n_agents(self):
        return self.env.n_agents

    @property
    def summary(self):
        last_completion_metrics = self.env.previous_episode_metrics
        if last_completion_metrics is None:
            return {}
        last_completion_metrics["completions"] = int(self.env._completions)
        return last_completion_metrics

    def get_seed(self):
        return int(self._seed)

    def seed(self, seed):
        # Using the seeding utility from OpenAI Gym
        # https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        _, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as an uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31

        if self.verbose:
            print(
                "[EnvWrapper] twisting seed {} -> {} -> {} (final)".format(
                    seed, seed1, seed2
                )
            )

        seed = int(seed2)
        np.random.seed(seed2)
        random.seed(seed2)
        self._seed = seed2

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return self.data_preprocess(recursive_list_to_np_array(obs))

    def step(self, action_dict):
        obs, rew, done, info = self.env.step(action_dict)
        assert isinstance(obs[self.sample_agent_idx]["action_mask"], np.ndarray)

        return self.data_preprocess(recursive_list_to_np_array(obs)), rew, done, info
