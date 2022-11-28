"""
PPO's top level algorithm.
Manages batching and multi-agent training.
"""
import copy
import logging
from multiprocessing import Pipe, Process
import sys

from algorithm.algorithm_config import AlgorithmConfig
from memory import BatchMemory
from policy.policy import PPOAgent
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
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
        # FIXME: doesn't work with `self.algorithm_config.policy_mapping_fun` reference
        self.memory = BatchMemory(
            self.algorithm_config.policy_mapping_function,
            self.algorithm_config.policies_configs,
            self.algorithm_config.agents_name,
        )

        if self.algorithm_config.multiprocessing:
            self.works, self.parent_conns, self.child_conns = [], [], []

            for idx in range(self.algorithm_config.num_workers):
                parent_conn, child_conn = Pipe()

                work = Environment(
                    env=self.algorithm_config.env,
                    seed=self.algorithm_config.seed + idx,
                    child_conn=child_conn
                )
                work.start()
                self.works.append(work)
                self.parent_conns.append(parent_conn)
                self.child_conns.append(child_conn)

            self.memory_dictionary = {}  # use to pass it
            # used for memory's step bf passing it to memory_dict
            self.batch_memory_dictionary = {}
            for idx, parent_conn in enumerate(self.parent_conns):
                self.batch_memory_dictionary[idx] = {"state": parent_conn.recv()}

                self.memory_dictionary[idx] = BatchMemory(
                    self.algorithm_config.policy_mapping_function,
                    self.algorithm_config.policies_configs,
                    self.algorithm_config.agents_name,
                )

    def kill_processes(self):
        for work in self.works:
            work.terminate()
            print("TERMINATED:", work)
            work.join()
    @timeit
    def batch_multi_process(self):
        for idx in range(self.algorithm_config.num_workers):
            self.memory_dictionary[idx].reset_memory()

        step = 0
        while (
            step < self.algorithm_config.batch_size // self.algorithm_config.num_workers
        ):
            # logging.debug(f"Batching step: {step}x{self.algorithm_config.num_workers}")
            for idx in range(self.algorithm_config.num_workers):
                (
                    self.batch_memory_dictionary[idx]["action"],
                    self.batch_memory_dictionary[idx]["action_onehot"],
                    self.batch_memory_dictionary[idx]["prediction"],
                ) = self.get_actions(self.batch_memory_dictionary[idx]["state"])
            for worker_id, parent_conn in enumerate(self.parent_conns):
                parent_conn.send(self.batch_memory_dictionary[worker_id]["action"])
            # Retrieve new state, rew
            for worker_id, parent_conn in enumerate(self.parent_conns):
                (
                    self.batch_memory_dictionary[worker_id]["next_state"],
                    self.batch_memory_dictionary[worker_id]["reward"],
                ) = parent_conn.recv()
            # next_state, reward, _, _ = env.step(action)

            # Memorize (state, action, reward) for trainig
            for idx in range(self.algorithm_config.num_workers):
                self.memory_dictionary
            # self.memory.update_memory(
            #     state, next_state, action_onehot, reward, prediction
            # )

            # update state for next step
            for idx in range(self.algorithm_config.num_workers):
                # state, next_state, action_onehot, reward, prediction
                self.memory_dictionary[idx].update_memory(
                    self.batch_memory_dictionary[idx]["state"],
                    self.batch_memory_dictionary[idx]["next_state"],
                    self.batch_memory_dictionary[idx]["action_onehot"],
                    self.batch_memory_dictionary[idx]["reward"],
                    self.batch_memory_dictionary[idx]["prediction"]
                )

                self.batch_memory_dictionary[idx][
                    "state"
                ] = self.batch_memory_dictionary[idx]["next_state"]

            step += 1
        
        # Get total memory
        print("getting memory")
        for idx in range(self.algorithm_config.num_workers):
            self.memory+=self.memory_dictionary[idx]

        # self.kill_processes()
        # sys.exit()
        

            

    @timeit
    def batch(self, env):
        logging.debug("Batching")
        state = env.reset()
        steps = 0
        while steps < self.algorithm_config.batch_size:
            if steps % 20 == 0:
                logging.debug(f"    step: {steps}")
            # Actor picks an action
            action, action_onehot, prediction = self.get_actions(state)

            # Retrieve new state, rew
            next_state, reward, _, _ = env.step(action)
            # print(f"            REWARD BATCH {reward}" )
            # Memorize (state, action, reward) for trainig
            self.memory.update_memory(
                state, next_state, action_onehot, reward, prediction
            )

            state = next_state
            steps += 1

    def train_one_step(
        self,
        env,
    ):
        """
        Train all Policys
        Here PPO's Minibatch is generated and splitted to each policy, following
        `policy_mapping_fun` rules
        """
        # Resetting memory
        self.memory.reset_memory()

        env = copy.deepcopy(env)

        # state = obs

        # Collecting data for batching
        if not self.algorithm_config.multiprocessing:
            self.batch(env)
        else:
            self.batch_multi_process()
        # sys.exit()
        # Pass batch to the correct policy to perform training
        for key in self.training_policies:
            logging.debug(f"Training policy {key}")
            self.training_policies[key].learn(*self.memory.get_memory(key))

    #@timeit
    def get_actions(self, obs: dict) -> dict:
        """
        Build action dictionary from env observations. Output has thi structure:

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

        actions, actions_onehot, predictions = {}, {}, {}
        for key in obs.keys():
            if key != "p":
                # print(self._policy_mapping_function(key))
                (
                    actions[key],
                    actions_onehot[key],
                    predictions[key],
                ) = self.training_policies[
                    self.algorithm_config.policy_mapping_function(key)
                ].act(
                    obs[key]
                )
            else:
                # tmp to also feed the planner
                actions[key], actions_onehot[key], predictions[key] = (
                    [0, 0, 0, 0, 0, 0, 0],
                    0,
                    0,
                )
        # logging.debug(actions)
        return actions, actions_onehot, predictions
