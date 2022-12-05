import copy
from dataclasses import dataclass
from functools import wraps
import time
import logging
import numpy as np
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


# @dataclass
class BatchMemory:
    def __init__(
        self,
        policy_mapping_function,
        policy_config: dict,
        available_agent_id: list,
        env,
    ):
        self.policy_mapping_function = policy_mapping_function
        # ['0','1','2','3','p']
        self.available_agent_ids = available_agent_id

        # Environment
        env = copy.deepcopy(env)
        obs = env.reset()

        def build_states(key):
            data_dict = {}
            for keys in obs[key].keys():
                data_dict[keys] = []
            return data_dict

        # states, next_states, actions, rewards, predictions, dones
        # self.keys = ["states", "next_states", "actions", "rewards", "predictions"]

        # key:dict
        self.states = {key: build_states(key) for key in self.available_agent_ids}
        # key:dict
        self.next_states = {key: build_states(key) for key in self.available_agent_ids}
        # key:list (onehot encoding)
        self.actions = {key: [] for key in self.available_agent_ids}
        # key:list
        self.rewards = {key: [] for key in self.available_agent_ids}
        # key:list
        self.predictions = {key: [] for key in self.available_agent_ids}

        """
        Actual structure is like this:
        self.states = {
            '0': {...},
            '1': {...},
            ...
            'p': {...},
        }
        self.next_states = {
            '0': {...},
            '1': {...},
            ...
            'p': {...},
        }
        ...
        self.predictions = {
            '0': {...},
            '1': {...},
            ...
            'p': {...},
        }
        """

    @timeit
    def reset_memory(self):
        """
        Clears the memory.
        """
        for key in self.available_agent_ids:
            self.actions[key].clear()
            self.rewards[key].clear()
            self.predictions[key].clear()

            for data in self.states[key].keys():
                self.states[key][data].clear()
                self.next_states[key][data].clear()

    def update_memory(
        self,
        state: dict,
        next_state: dict,
        action_onehot: dict,
        reward: dict,
        prediction: dict,
    ):
        """ """
        for key in self.available_agent_ids:
            self.actions[key].append(action_onehot[key])
            self.rewards[key].append(reward[key])
            self.predictions[key].append(prediction[key])

            for data in self.states[key].keys():
                self.states[key][data].append(state[key][data])
                self.next_states[key][data].append(next_state[key][data])

    @timeit
    @tf.function
    def get_memory(self, mapped_key):
        this_state, this_next_state, this_action, this_reward, this_prediction = (
            {},
            {},
            [],
            [],
            [],
        )

        if mapped_key == "a":
            # Build empty this_state, this_next_state
            for item in ["world-map", "world-idx_map", "time", "flat", "action_mask"]:
                this_state[item] = []
                this_next_state[item] = []
        elif mapped_key == "p":
            for item in ["world-map", "world-idx_map", "time", "flat", "p0", "p1", "p2", "p3", "action_mask"]:
                this_state[item] = []
                this_next_state[item] = []
        else:
            KeyError(f"Key {mapped_key} is not registered in the training cycle.")

        for key in self.available_agent_ids:
            if self.policy_mapping_function(key) == mapped_key:
                for data in self.states[key].keys():
                    this_state[data].extend(self.states[key][data])
                    this_next_state[data].extend(self.next_states[key][data])

                this_action.extend(self.actions[key])
                this_reward.extend(self.rewards[key])
                this_prediction.extend(self.predictions[key])

        for key in this_state.keys():
            this_state[key]=np.array(this_state[key])
            this_next_state[key]=np.array(this_next_state[key])

        return this_state, np.array(this_action), np.array(this_reward), np.array(this_prediction), this_next_state
