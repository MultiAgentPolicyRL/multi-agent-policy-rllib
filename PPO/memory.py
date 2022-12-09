import copy
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
        self.observation = {key: build_states(key) for key in self.available_agent_ids}
        # key:dict
        self.next_observation = {key: build_states(key) for key in self.available_agent_ids}
        # key:list (onehot encoding)
        self.policy_actions = {key: [] for key in self.available_agent_ids}
        # key:list
        self.rewards = {key: [] for key in self.available_agent_ids}
        # key:list
        
        self.vf_predictions = {key: [] for key in self.available_agent_ids}
        self.vf_predictions_old = {key: [] for key in self.available_agent_ids}

        # values, states_h_p, states_c_p, states_h_v, states_c_v
        self.states_h_p = {key: [] for key in self.available_agent_ids}
        self.states_c_p = {key: [] for key in self.available_agent_ids}
        self.states_h_v = {key: [] for key in self.available_agent_ids}
        self.states_c_v = {key: [] for key in self.available_agent_ids}
        
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
            self.policy_actions[key].clear()
            self.rewards[key].clear()
            self.vf_predictions[key].clear()
            self.vf_predictions_old[key].clear()

            self.states_h_p[key].clear()
            self.states_c_p[key].clear()
            self.states_h_v[key].clear()
            self.states_c_v[key].clear()

            for data in self.observation[key].keys():
                self.observation[key][data].clear()
                self.next_observation[key][data].clear()

    def update_memory(
        self,
        observation: dict,
        next_observation: dict,
        policy_action_onehot: dict,
        vf_prediction: dict,
        vf_prediction_old:dict,
        reward: dict,

        states_h_p: dict,
        states_c_p: dict,
        states_h_v: dict,
        states_c_v: dict,
    ):
        """ 
        Updates memory

        Args:
            observation, 
            next_observation, 
            policy_action_onehot, 
            vf_prediction, 
            predictions_old, 
            reward, 

            states_h_p, 
            states_c_p, 
            states_h_v, 
            states_c_v, 

        """
        for key in self.available_agent_ids:
            self.policy_actions[key].append(policy_action_onehot[key])
            self.rewards[key].append(reward[key])
            self.vf_predictions[key].append(vf_prediction[key])
            self.vf_predictions_old[key].append(vf_prediction_old[key])
            if key != 'p':

                self.states_h_p[key].append(states_h_p[key])
                self.states_c_p[key].append(states_c_p[key])
                self.states_h_v[key].append(states_h_v[key])
                self.states_c_v[key].append(states_c_v[key])


            for data in self.observation[key].keys():
                self.observation[key][data].append(observation[key][data])
                self.next_observation[key][data].append(next_observation[key][data])

    @timeit
    # @tf.function
    def get_memory(self, mapped_key):
        """
        returns:
            observation, next_observation, policy_action, vf_prediction, vf_prediction_old, reward, states_h_p, states_c_p, states_h_v, states_c_v 
        """
        this_observation = {}
        this_next_observation = {}
        this_policy_action = []
        this_reward = []
        
        this_vf_prediction = []
        this_vf_prediction_old = []

        this_states_h_p = []
        this_states_c_p = []
        this_states_h_v = []
        this_states_c_v = []

        if mapped_key == "a":
            # Build empty this_state, this_next_state
            for item in ["world-map", "world-idx_map", "time", "flat", "action_mask"]:
                this_observation[item] = []
                this_next_observation[item] = []
        elif mapped_key == "p":
            for item in [
                "world-map",
                "world-idx_map",
                "time",
                "flat",
                "p0",
                "p1",
                "p2",
                "p3",
                "action_mask",
            ]:
                this_observation[item] = []
                this_next_observation[item] = []
        else:
            KeyError(f"Key {mapped_key} is not registered in the training cycle.")

        for key in self.available_agent_ids:
            if self.policy_mapping_function(key) == mapped_key:
                for data in self.observation[key].keys():
                    this_observation[data].extend(self.observation[key][data])
                    this_next_observation[data].extend(self.next_observation[key][data])

                this_policy_action.extend(self.policy_actions[key])
                this_reward.extend(self.rewards[key])
                this_vf_prediction.extend(self.vf_predictions[key])
                this_vf_prediction_old.extend(self.vf_predictions_old[key])
                
                this_states_h_p.extend(self.states_h_p[key])
                this_states_c_p.extend(self.states_c_p[key])
                this_states_h_v.extend(self.states_h_v[key])
                this_states_c_v.extend(self.states_c_v[key])


        for key in this_observation.keys():
            this_observation[key] = np.array(this_observation[key])
            this_next_observation[key] = np.array(this_next_observation[key])

        return (
            this_observation,
            this_next_observation,
            np.array(this_policy_action),
            np.array(this_vf_prediction),
            np.array(this_vf_prediction_old),
            np.array(this_reward),

            np.array(this_states_h_p),
            np.array(this_states_c_p),
            np.array(this_states_h_v),
            np.array(this_states_c_v),
        )
