from dataclasses import dataclass
from functools import wraps
import time
import logging
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
class BatchMemory():
    def __init__(
        self, policy_mapping_function, policy_config: dict, available_agent_id: list
    ):
        self.policy_mapping_function = policy_mapping_function
        # ['0','1','2','3','p']
        self.available_agent_id = available_agent_id

        # states, next_states, actions, rewards, predictions, dones
        self.keys = ["states", "next_states", "actions", "rewards", "predictions"]

        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.predictions = []

        # Initialize empty memory
        # for key in available_agent_groups:
        #     self.batch[key] = {e: [] for e in keys}
    # @timeit
    def __add__(self, other):
        self.states += other.states
        self.next_states += other.next_states
        self.actions += other.actions
        self.rewards += other.rewards
        self.predictions += other.predictions
        return self

    @timeit
    def reset_memory(self):
        """
        Method.
        Clears the memory.
        """
        self.states.clear()
        self.next_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.predictions.clear()

    #@timeit
    def update_memory(
        self,
        state: dict,
        next_state: dict,
        action_onehot: dict,
        reward: dict,
        prediction: dict,
    ):
        """ """
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action_onehot)
        self.rewards.append(reward)
        self.predictions.append(prediction)

    @timeit
    # @tf.function
    def get_memory(self, key):
        data_structure = {}
        # keys = ['0','1','2','3','p']
        for keys in self.available_agent_id:
            data_structure[keys] = {
                "state": [],
                "next_state": [],
                "action": [],
                "reward": [],
                "prediction": [],
            }

        for state, next_state, action, reward, prediction in zip(
            self.states, self.next_states, self.actions, self.rewards, self.predictions
        ):
            for keys in self.available_agent_id:
                data_structure[keys]["state"].append(state[keys])
                data_structure[keys]["next_state"].append(next_state[keys])
                data_structure[keys]["action"].append(action[keys])
                data_structure[keys]["reward"].append(reward[keys])
                data_structure[keys]["prediction"].append(prediction[keys])

        this_state, this_next_state, this_action, this_reward, this_prediction = (
            [],
            [],
            [],
            [],
            [],
        )

        for keys in data_structure.keys():
            if self.policy_mapping_function(keys) == key:
                for state, next_state, action, reward, prediction in zip(
                    data_structure[keys]["state"],
                    data_structure[keys]["next_state"],
                    data_structure[keys]["action"],
                    data_structure[keys]["reward"],
                    data_structure[keys]["prediction"],
                ):
                    this_state.append(state)
                    this_next_state.append(next_state)
                    this_action.append(action)
                    this_reward.append(reward)
                    this_prediction.append(prediction)

        return this_state, this_action, this_reward, this_prediction, this_next_state
