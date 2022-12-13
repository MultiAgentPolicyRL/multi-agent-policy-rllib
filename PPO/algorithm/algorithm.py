"""
PPO's top level algorithm.
Manages batching and multi-agent training.
"""
import copy
import logging
from multiprocessing import Pipe, Process
import sys

import numpy as np
from algorithm.algorithm_config import AlgorithmConfig
from memory import BatchMemory
from policy.policy import PPOAgent
from functools import wraps
import time
import tensorflow as tf


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logging.debug(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


class Environment(Process):
    def __init__(self, env, seed, child_conn):
        super(Environment, self).__init__()
        self.env = copy.deepcopy(env)
        self.env.seed(seed)
        self.child_conn = child_conn
        self.obs = self.env.reset()

    def run(self):
        super(Environment, self).run()
        self.child_conn.send(self.obs)

        while True:
            action = self.child_conn.recv()

            state, reward, _, _ = self.env.step(action)

            self.child_conn.send([state, reward])


class PpoAlgorithm(object):
    """
    PPO's top level algorithm.
    Manages batching and multi-agent training.
    """

    def __init__(self, algorithm_config: AlgorithmConfig):
        """
        policy_config: dict,
        available_agent_groups,
        policy_mapping_fun=None

        policy config can be like:

        policy_config: {
            'a' : config or None,
            'p' : config or None,
        }
        """
        self.algorithm_config = algorithm_config

        ###
        # Build dictionary for each policy (key) in `policy_config`
        ###
        self.training_policies = {}
        for key in self.algorithm_config.policies_configs:
            # config injection

            self.training_policies[key]: PPOAgent = PPOAgent(
                self.algorithm_config.policies_configs[key]
            )

        # Setup batch memory
        self.memory = BatchMemory(
            self.algorithm_config.policy_mapping_function,
            self.algorithm_config.policies_configs,
            self.algorithm_config.agents_name,
            self.algorithm_config.env,
        )

    @timeit
    def _batch(self, env, states_h_p, states_c_p, states_h_v, states_c_v):
        # logging.debug("Batching")
        state = env.reset()
        steps = 0
        vf_prediction_old = {
            '0': np.array([0.0]),
            '1': np.array([0.0]),
            '2': np.array([0.0]),
            '3': np.array([0.0]),
            'p': np.array([0.0])
        }
        

        while steps < self.algorithm_config.batch_size:
            # if steps % 100 == 0:
            #     logging.debug(f"    step: {steps}")
            # Actor picks an action
            # obs: dict, seq_in: Any, state_in_h_p: dict, state_in_c_p: dict, state_in_h_v: dict, state_in_c_v: dict
            (
                policy_actions,
                policy_actions_onehot,
                policy_predictions,
                vf_predictions,
                states_h_p,
                states_c_p,
                states_h_v,
                states_c_v,
            ) = self.get_actions(state, 1, states_h_p, states_c_p, states_h_v, states_c_v)

            # Retrieve new state, rew
            next_state, reward, _, _ = env.step(policy_actions)
            # print(f"            REWARD BATCH {reward}" )
            # Memorize (state, action, reward) for trainig
            self.memory.update_memory(
                # observation, next_observation, policy_action_onehot, vf_prediction, predictions_old, reward,
                #  values, states_h_p, states_c_p, states_h_v, states_c_v,
                observation=state,
                next_observation=next_state,
                policy_action_onehot=policy_actions_onehot,
                vf_prediction=vf_predictions,
                vf_prediction_old=vf_prediction_old,
                reward=reward,
                states_h_p=states_h_p, 
                states_c_p=states_c_p,
                states_h_v=states_h_v, 
                states_c_v=states_c_v,
            )

            state = next_state
            vf_prediction_old = vf_predictions
            steps += 1

    def train_one_step(
        self,
        env,
        states_h_p, states_c_p, states_h_v, states_c_v
    ):
        """
        Train all Policys
        Here PPO's Minibatch is generated and splitted to each policy, following
        `policy_mapping_fun` rules
        """
        # Resetting memory
        self.memory.reset_memory()
        env = copy.deepcopy(env)

        # Collecting data for batching
        self._batch(env,states_h_p, states_c_p, states_h_v, states_c_v)

        # sys.exit("algorithm line 130")
        # Pass batch to the correct policy to perform training
        for key in self.training_policies:
            # logging.debug(f"Training policy {key}")
            self.training_policies[key].learn(*self.memory.get_memory(key))

    # @timeit
    def get_actions(
        self,
        obs: dict,
        seq_in,
        state_in_h_p: dict,
        state_in_c_p: dict,
        state_in_h_v: dict,
        state_in_c_v: dict,
    ) -> dict:
        """


        Args:
            obs:
            seq_in:
            state_in_h_p:
            state_in_c_p:
            state_in_h_v:
            state_in_c_v:

        Returns:
            policy_actions,
            policy_actions_onehot,
            policy_predictions,
            vf_predictions,
            states_h_p,
            states_c_p,
            states_h_v,
            states_c_v,


        Build action dictionary from env observations. 
        
        
        Output has this structure:

                actions: {
                    '0': [...],
                    '1': [...],
                    '2': [...],
                    ...
                    'p': [...]
                }

        FIXME: Planner
        FIXME: TOO SLOW!!!!!

        Arguments:
            obs: observation dictionary of the environment, it contains all observations for each agent

        Returns:
            actions dict: actions for each agent
        """

        (
            policy_actions,
            policy_actions_onehot,
            policy_predictions,
            vf_predictions,
            states_h_p,
            states_c_p,
            states_h_v,
            states_c_v,
        ) = ({}, {}, {}, {}, {}, {}, {}, {})
        for key in obs.keys():
            if key != "p":

                # inputs, seq_in, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v
                # outputs: policy_action[0], policy_action_onehot, policy_prediction, vf_prediction, state_h_p, state_c_p, state_h_v, state_c_v

                (
                    policy_actions[key],
                    policy_actions_onehot[key],
                    policy_predictions[key],
                    vf_predictions[key],
                    states_h_p[key],
                    states_c_p[key],
                    states_h_v[key],
                    states_c_v[key],
                ) = self.training_policies[
                    self.algorithm_config.policy_mapping_function(key)
                ].act(
                    obs[key],
                    seq_in,
                    state_in_h_p[key],
                    state_in_c_p[key],
                    state_in_h_v[key],
                    state_in_c_v[key],
                )
            else:
                # tmp to also feed the planner
                policy_actions[key], policy_actions_onehot[key], policy_predictions[key], vf_predictions[key] = (
                    [0, 0, 0, 0, 0, 0, 0],
                    0,
                    0,
                    0,
                )
        # logging.debug(actions)
        return (
            policy_actions,
            policy_actions_onehot,
            policy_predictions,
            vf_predictions,
            states_h_p,
            states_c_p,
            states_h_v,
            states_c_v,
        )
