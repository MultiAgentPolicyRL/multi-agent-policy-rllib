"""
docs about this file
"""
import copy
from functools import wraps
import logging
import random
import sys

import numpy as np
import tensorflow as tf
from deprecated import deprecated
# from model.model import ActorModel, CriticModel
from model.new_model import Model
from policy.policy_config import PolicyConfig
import time


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

class PPOAgent:
    """
    PPO Main Optimization Algorithm
    """

    def __init__(self, policy_config: PolicyConfig):
        # Initialization
        # Environment and PPO parameters
        self.policy_config = policy_config
        self.action_space = self.policy_config.action_space  # self.env.action_space.n
        self.batch_size = self.policy_config.batch_size  # training epochs

        # Create Actor-Critic network model
        self.Model = Model(policy_config.model_config)

    def act(self, state, seq_in, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v):
        """
        seq_in,
                    state_in_h_p[key],
                    state_in_c_p[key],
                    state_in_h_v[key],
                    state_in_c_v[key],
        FIXME:1
        FIXME: if we can use tf instead of np this function can be @tf.function-ed
        example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        # Use the network to predict the next action to take, using the model
        # [logits, values, state_h_p, state_c_p, state_h_v, state_c_v]
        policy_prediction, vf_prediction, state_h_p, state_c_p, state_h_v, state_c_v = self.Model(
            state, seq_in, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v
            )
        
        print(vf_prediction)
        policy_prediction = np.squeeze(policy_prediction.numpy())
        
        # prediction = np.where(prediction != -1.0*pow(10, 7), prediction, 0 )
        
        # prediction = prediction/np.sum(prediction)
        # print(prediction)

        policy_action = random.choices(np.arange(50), weights=policy_prediction)
        # action = np.random.choice(np.arange(50), p=prediction)
        policy_action_onehot = np.zeros([self.action_space])
        policy_action_onehot[policy_action] = 1

        return policy_action[0], policy_action_onehot, policy_prediction, vf_prediction, state_h_p, state_c_p, state_h_v, state_c_v

    def _get_gaes(
        self,
        rewards,
        values,
        next_values,
        gamma=0.998,
        lamda=0.98,
        normalize=True,
    ):
        """
        # FIXME: improve excecution time.
        Gae's calculation
        Removed dones
        """
        deltas = [r + gamma * nv - v for r, nv, v in zip(rewards, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)

        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + gamma * lamda * gaes[t + 1]

        target = gaes + values

        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return np.vstack(gaes), np.vstack(target)

    def learn(
        # observation, 
        # next_observation, 
        # policy_action, 
        # vf_prediction,
        # vf_prediction_old, 
        # reward, 
        # states_h_p, 
        # states_c_p, 
        # states_h_v, 
        # states_c_v 

        self,
        observation: dict,
        next_observation: dict,
        policy_action: list,
        vf_predictions: list,
        vf_predictions_old,
        reward: list,
        states_h_p,
        states_c_p,
        states_h_v,
        states_c_v,
    ):
        """
        Train Policy networks
        """
        # # Get Critic network predictions
        # tempo = time.time()
        # values = self.Critic.batch_predict(states)
        # next_values = self.Critic.batch_predict(next_states)
        # logging.debug(f"     Values and next_values required {time.time()-tempo}s")

        # Compute discounted rewards and advantages
        # GAE
        tempo = time.time()

        # print(f"VF PREDICTIONS OLD: {np.squeeze(vf_predictions_old)}")
        # print(f"VF PREDICTIONS {np.squeeze(vf_predictions)}")

        advantages, target = self._get_gaes(
            np.array(reward), np.squeeze(np.squeeze(vf_predictions_old)), np.squeeze(np.squeeze(vf_predictions))
        )

        # print(advantages.shape)
        # print(np.squeeze(vf_predictions, (1,2)).shape)
        # print(policy_action.shape)
        # print(target.shape)
        # sys.exit("POLicY riga 154")

        # print(advantages)
        # print(target)

        # sys.exit("POLicY riga 154")


        logging.debug(f"     Gaes required {time.time()-tempo}s")

        tempo = time.time()
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        y_true = np.hstack([advantages, np.squeeze(vf_predictions, (1,2)), policy_action, target])

        logging.debug(f"     Data prep required: {time.time()-tempo}s")
        tempo = time.time()

        # obs[key],
        # seq_in,
        # state_in_h_p[key],
        # state_in_c_p[key],
        # state_in_h_v[key],
        # state_in_c_v[key],

        # training Actor and Critic networks
        a_loss = self.Model.model.fit(
            x=[observation, 2, states_h_p, states_c_p, states_h_v, states_c_v],
            y=y_true,
            epochs=self.policy_config.agents_per_possible_policy
            * self.policy_config.num_workers,
            steps_per_epoch=self.batch_size // self.policy_config.num_workers,
            verbose=0,
            # shuffle=self.shuffle,
            workers=8,
            use_multiprocessing=True,
        )

        logging.debug(f"     Fit Actor Network required {time.time()-tempo}s")
        logging.debug(f"        Actor loss: {a_loss.history['loss'][-1]}")
        sys.exit("POLicY riga 154")

        # tempo = time.time()
        # c_loss = self.Critic.critic.fit(
        #     x=[states["world-map"], states["flat"]],
        #     y=target,
        #     epochs=1,
        #     steps_per_epoch=self.batch_size,
        #     verbose=0,
        #     # shuffle=self.shuffle,
        #     workers=8,
        #     use_multiprocessing=True,
        # )
        # logging.debug(f"     Fit Critic Network required {time.time()-tempo}s")

        # logging.debug(f"        Critic loss: {c_loss.history['loss'][-1]}")

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

    def _policy_mapping_fun(self, i: str) -> str:
        """
        Use it by passing keys of a dictionary to differentiate between agents

        default for ai-economist environment:
        returns a if `i` is a number -> if the key of the dictionary is a number,
        returns p if `i` is a string -> social planner
        """
        if str(i).isdigit() or i == "a":
            return "a"
        return "p"

    # @deprecated
    # def build_action_dict(self, obs: dict):
    #     """


    #     Build an action dictionary that can be used in training
    #     FIXME: right now, for developing reasons `p`'s policy doesn't exist and is not manged:
    #     so paller's action will be `0`

    #     Arguments:
    #         obs: environment observations

    #     Returns:
    #         A dictionary containing an action for each agent
    #     """
    #     actions = {}
    #     actions_oneshot = {}
    #     predictions = {}

    #     for key in obs.keys():
    #         if self._policy_mapping_fun(key) == "a":
    #             actions[key], actions_oneshot[key], predictions[key] = self.act(
    #                 obs[key]
    #             )
    #         elif self._policy_mapping_fun(key) == "p":
    #             actions["p"] = [0, 0, 0, 0, 0, 0, 0]
    #         else:
    #             IndexError(f"this actor is not managed by the environment, key: {key}")

    #     return actions, actions_oneshot, predictions

    # @deprecated
    # def train_one_step_with_batch(self, data):
    #     """
    #     Train agents for one step using mini_batching
    #     """
    #     data.batch
    #     self.learn()
