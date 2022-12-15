"""
docs about this file
"""
import copy
import logging
import random
import sys
import time

import numpy as np

# import tensorflow as tf
import torch
from model.model import LSTMModel
from PPO.policy.ppo_policy_config import PpoPolicyConfig
from utils.timeit import timeit


class PPOAgent:
    """
    PPO Main Optimization Algorithm
    """

    def __init__(self, policy_config: PpoPolicyConfig):
        # Initialization
        # Environment and PPO parameters
        self.policy_config = policy_config
        self.action_space = self.policy_config.action_space  # self.env.action_space.n
        self.batch_size = self.policy_config.batch_size  # training epochs

        self.Model: LSTMModel = LSTMModel(policy_config.model_config)

    @timeit
    def act(self, observation: dict):
        """
        Given an observation, returns `policy_action`, `policy_probability` and `vf_action` from the model.
        In this case (PPO) it's just a reference to call the model's forward method ->
        it's an "exposed API": common named functions for each policy.

        Args:
            observation: single agent observation of the environment.

        Returns:
            policy_action: predicted action(s)
            policy_probability: action probabilities
            vf_action: value function action predicted
        """
        # torch.squeeze(logits)

        # Get the prediction from the Actor network
        with torch.no_grad():
            policy_action, policy_probability, vf_action = self.Model(
                observation)

        return policy_action, policy_probability, vf_action

    def learn(
        self,
        observations: list,
        policy_actions: list,
        policy_probabilitiess: list,
        value_functions: list,
        rewards: list,
        epochs: int,
        steps_per_epoch: int
    ):
        """
        Train Policy networks
        Takes as input the batch with N epochs of M steps_per_epoch. As we are using an LSTM
        model we are not shuffling all the data to create the minibatch, but only shuffling
        each epoch.

        Example:
            Input epochs: 0,1,2,3
            Shuffled epochs: 2,0,1,3

        It calls `self.Model.fit` passing the shuffled epoch.

        INFO: edit this function to modify how minibatch creation is managed.
        INFO: there's a simpler way to do the same, but memory need to be adapted for it to work.

            data = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
            epochs = 4
            steps_per_epoch = 5
            for i in range(epochs):
                print(dati[:i:steps_per_epoch])

        Args:
            observations: Agent ordered, time listed observations per agent
            policy_actions: Agent ordered, time listed policy_actions per agent
            policy_probabilitiess: Agent ordered, time listed policy_probabilitiess per agent
            value_functions: Agent ordered, time listed observalue_functionsvations per agent
            rewards: Agent ordered, time listed rewards per agent
            epochs: how many epochs in the given batch (it is equal to n_agents in the selected
            batch)
            steps_per_epoch: how long is the epoch (it's equal to algorithm_config.batch_size)

        Returns:
            nothing

        """
    
        
        """
        Logic simplified:
            epochs = 4
            batch_size = 5
            epochs_selected = list(range(epochs))
            random.shuffle(epochs_selected)
            data = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3]
            for i in epochs_selected:
                print(data[i*batch_size:batch_size+i*batch_size])
        """
        # Set epochs order
        epochs_order = list(range(epochs))
        random.shuffle(epochs_order)

        for i in epochs_order:
            # Get it's data
            selected_observations = observations[i*steps_per_epoch:steps_per_epoch+i*steps_per_epoch]
            selected_policy_actions =  policy_actions[i*steps_per_epoch:steps_per_epoch+i*steps_per_epoch]
            selected_policy_probabilitiess =  policy_probabilitiess[i*steps_per_epoch:steps_per_epoch+i*steps_per_epoch]
            selected_value_functions =  value_functions[i*steps_per_epoch:steps_per_epoch+i*steps_per_epoch]
            selected_rewards =  rewards[i*steps_per_epoch:steps_per_epoch+i*steps_per_epoch]

            # Fit the model
            self.Model.fit(
                observations = selected_observations,
                policy_actions =selected_policy_actions,
                policy_probabilitiess = selected_policy_probabilitiess,
                value_functions = selected_value_functions,
                rewards = selected_rewards
            )





        # y_true = [advantages, vf_predictions, policy_actions, target]

        # # FIT
        # self.Model.fit(observations, y_true)

        # values = self.Critic.batch_predict(observations)
        # next_values = self.Critic.batch_predict(next_observations)

        # logging.debug(f"     Values and next_values required {time.time()-tempo}s")

        # # Compute discounted rewards and advantages
        # # GAE
        # tempo = time.time()

        # advantages, target = self._get_gaes(
        #     rewards, np.squeeze(values), np.squeeze(next_values)
        # )

        # logging.debug(f"     Gaes required {time.time()-tempo}s")

        # # stack everything to numpy array
        # # pack all advantages, predictions and actions to y_true and when they are received
        # # in custom PPO loss function we unpack it
        # tempo = time.time()
        # y_true = np.hstack([advantages, vf_predictions, policy_actions])
        # logging.debug(f"     Data prep required: {time.time()-tempo}s")

        # tempo = time.time()

        # # training Actor and Critic networks
        # a_loss = self.Actor.actor.fit(
        #     [world_map, flat],
        #     y_true,
        #     # batch_size=self.batch_size,
        #     epochs=self.policy_config.agents_per_possible_policy
        #     * self.policy_config.num_workers,
        #     steps_per_epoch=self.batch_size // self.policy_config.num_workers,
        #     verbose=0,
        #     shuffle=self.shuffle,
        #     workers=8,
        #     use_multiprocessing=True,
        # )
        # logging.debug(f"     Fit Actor Network required {time.time()-tempo}s")
        # logging.debug(f"        Actor loss: {a_loss.history['loss'][-1]}")

        # tempo = time.time()
        # values = tf.convert_to_tensor(values)
        # target = [target, values]
        # logging.debug(f"    Prep 2 required {time.time()-tempo}")

        # tempo = time.time()
        # c_loss = self.Critic.critic.fit(
        #     [world_map, flat],
        #     target,
        #     # batch_size=self.batch_size,
        #     epochs=1,
        #     steps_per_epoch=self.batch_size,
        #     verbose=0,
        #     shuffle=self.shuffle,
        #     workers=8,
        #     use_multiprocessing=True,
        # )
        # logging.debug(f"     Fit Critic Network required {time.time()-tempo}s")

        # logging.debug(f"        Critic loss: {c_loss.history['loss'][-1]}")

    # def _obs_dict_to_tensor_list(self, observation: dict):
    #     """
    #     Converts a dict of numpy.ndarrays to torch.tensors

    #     Args:
    #         observation: Single agent environment observation
    #     """
    #     output = []
    #     for key in observation.keys():
    #         output.append(torch.FloatTensor(observation[key]).unsqueeze(0)) # pylint: disable=no-member

    #     return output

    # # @timeit
    # def _get_gaes(
    #     self,
    #     rewards,
    #     values,
    #     next_values,
    #     gamma=0.998,
    #     lamda=0.98,
    #     normalize=True,
    # ):
    #     """
    #     Gae's calculation
    #     Removed dones
    #     """
    #     deltas = [r + gamma * nv - v for r, nv, v in zip(rewards, next_values, values)]
    #     deltas = np.stack(deltas)
    #     gaes = copy.deepcopy(deltas)

    #     for t in reversed(range(len(deltas) - 1)):
    #         gaes[t] = gaes[t] + gamma * lamda * gaes[t + 1]

    #     target = gaes + values

    #     if normalize:
    #         gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

    #     return np.vstack(gaes), np.vstack(target)

    # def _load(self) -> None:
    #     """
    #     Save Actor and Critic weights'
    #     """
    #     self.Actor.actor.load_weights(self.Actor_name)
    #     self.Critic.critic.load_weights(self.Critic_name)

    # def _save(self) -> None:
    #     """
    #     Load Actor and Critic weights'
    #     """
    #     self.Actor.actor.save_weights(self.Actor_name)
    #     self.Critic.critic.save_weights(self.Critic_name)

    # def _policy_mapping_fun(self, i: str) -> str:
    #     """
    #     Use it by passing keys of a dictionary to differentiate between agents

    #     default for ai-economist environment:
    #     returns a if `i` is a number -> if the key of the dictionary is a number,
    #     returns p if `i` is a string -> social planner
    #     """
    #     if str(i).isdigit() or i == "a":
    #         return "a"
    #     return "p"
