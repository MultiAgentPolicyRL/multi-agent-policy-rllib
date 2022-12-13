import copy
import sys
from dataclasses import dataclass

import torch
from utils.timeit import timeit
import numpy as np


@dataclass
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

        # state, next_state, action_onehot, reward, actor_prediction, vf_prediction, vf_prediction_old
        # self.keys = ["states", "next_states", "actions", "rewards", "predictions"]

        # key:dict -= state
        self.observations = {key: build_states(key) for key in self.available_agent_ids}
        # key:dict - next_state
        self.next_observations = {
            key: build_states(key) for key in self.available_agent_ids
        }
        # key:list (onehot encoding) - action_onehot
        self.policy_actions = {key: [] for key in self.available_agent_ids}
        # key:list
        self.policy_predictions = {key: [] for key in self.available_agent_ids}
        # key:list - reward
        self.rewards = {key: [] for key in self.available_agent_ids}
        self.vf_predictions = {key: [] for key in self.available_agent_ids}
        self.vf_predictions_old = {key: [] for key in self.available_agent_ids}

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
            self.policy_predictions[key].clear()
            self.vf_predictions[key].clear()
            self.vf_predictions_old[key].clear()

            for data in self.observations[key].keys():
                self.observations[key][data].clear()
                self.next_observations[key][data].clear()

    # @timeit
    def update_memory(
        self,
        observation: dict,
        next_observation: dict,
        policy_action_onehot: dict,
        reward: dict,
        policy_prediction: dict,
        vf_prediction: dict,
        vf_prediction_old: dict,
    ):
        """
        Updates memory
        """
        for key in self.available_agent_ids:
            self.policy_actions[key].append(policy_action_onehot[key].numpy())
            self.rewards[key].append(reward[key])
            self.policy_predictions[key].append(policy_prediction[key].numpy())
            self.vf_predictions[key].append(vf_prediction[key])
            self.vf_predictions_old[key].append(vf_prediction_old[key])

            for data in self.observations[key].keys():
                self.observations[key][data].append(observation[key][data])
                self.next_observations[key][data].append(next_observation[key][data])

    @timeit
    def get_memory(self, mapped_key):
        """
        Returns:
            observation,
            next_observation,
            policy_action
            policy_predictions
            reward
            vf_prediction
            vf_prediction_old
        """

        this_observation = {}
        this_next_observation = {}
        this_new_observation = {}

        this_policy_action = []
        this_reward = []
        this_policy_predictions = []
        this_vf_prediction = []
        this_vf_prediction_old = []

        if mapped_key == "a":
            # Build empty this_state, this_next_state
            for item in ["world-map", "world-idx_map", "time", "flat", "action_mask"]:
                this_observation[item] = []
                this_next_observation[item] = []
                this_new_observation[item] = []
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
                this_new_observation[item] = []

        else:
            KeyError(f"Key {mapped_key} is not registered in the training cycle.")

        for key in self.available_agent_ids:
            if self.policy_mapping_function(key) == mapped_key:
                for data in self.observations[key].keys():
                    this_observation[data].extend(self.observations[key][data])
                    this_next_observation[data].extend(
                        self.next_observations[key][data]
                    )

                this_policy_action.extend(self.policy_actions[key])
                this_reward.extend(self.rewards[key])
                this_policy_predictions.extend(self.policy_predictions[key])
                this_vf_prediction.extend(self.vf_predictions[key])
                this_vf_prediction_old.extend(self.vf_predictions_old[key])

        for key in this_observation.keys():
            this_observation[key] = np.array(this_observation[key])
            this_next_observation[key] = np.array(this_next_observation[key])

        # TESTING STUFF
        for key in this_observation.keys():
            for item in this_observation[key]:
                this_new_observation[key].extend(torch.FloatTensor(item).unsqueeze(0))


        return (
            this_new_observation,
            this_next_observation,
            np.array(this_policy_action),
            this_policy_predictions,
            np.array(this_reward),
            np.array(this_vf_prediction, dtype=object),
            np.array(this_vf_prediction_old, dtype=object),
        )
