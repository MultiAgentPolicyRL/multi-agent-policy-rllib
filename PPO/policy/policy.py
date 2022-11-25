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
from model.model import ActorModel, CriticModel
from policy.policy_config import PolicyConfig
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


# @tf.function(jit_compile=True)


class PPOAgent:
    """
    PPO Main Optimization Algorithm
    """

    def __init__(self, policy_config: PolicyConfig):
        # Initialization
        # Environment and PPO parameters
        self.policy_config = policy_config
        self.action_space = self.policy_config.action_space  # self.env.action_space.n
        self.max_average = 0  # when average score is above 0 model will be saved
        self.batch_size = self.policy_config.batch_size  # training epochs
        self.shuffle = False

        # if policy_config is not None:
        #     self.action_space = policy_config["action_space"]
        #     self.observation_space:  gym.spaces = policy_config["observation_space"]

        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = (
            [],
            [],
            [],
        )  # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = ActorModel(policy_config.model_config)
        self.Critic = CriticModel(policy_config.model_config)

        # self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        # self.Critic_name = f"{self.env_name}_PPO_Critic.h5"

    # @timeit
    def act(self, state):
        """
        No idea why with numpy isnt working.

        example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        # Use the network to predict the next action to take, using the model
        # start_time = time.perf_counter()
        prediction = self.Actor.predict(state)
        # print(f"    PREDICTION TIME: {time.perf_counter()-start_time}")
        # logging.debug(f"ACTING AAAAA {prediction}")
        # action = int(random.choices(state["action_mask"], weights=prediction)[0])
        # start_time = time.perf_counter()
        action = int(random.choices(np.arange(50), weights=prediction)[0])
        # print(f"    RANDOM CHOICES TIME: {time.perf_counter()-start_time}")
        # action = np.random.choice(self.action_size, p=prediction)
        # start_time = time.perf_counter()
        action_onehot = np.zeros([self.action_space])
        # print(f"    NP ZEROS TIME: {time.perf_counter()-start_time}")
        action_onehot[action] = 1

        return action, action_onehot, prediction

    # @timeit
    # @tf.function
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

        return np.vstack(gaes), tf.convert_to_tensor(np.vstack(target))

    def learn(
        self,
        states: list,
        actions: list,
        rewards: list,
        predictions: list,
        next_states: list,
    ):
        """
        Train Policy networks
        """
        # Get Critic network predictions
        values = self.Critic.batch_predict(np.array(states))
        next_values = self.Critic.batch_predict(next_states)

        # Compute discounted rewards and advantages
        # GAE
        logging.debug("     Calculating gaes")
        advantages, target = self._get_gaes(
            rewards, np.squeeze(values), np.squeeze(next_values)
        )

        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        y_true = np.hstack([advantages, predictions, actions])
        # print(y_true.shape)                       (n_agents*steps, 101) -> 101 = 1+50+50
        # print(advantages.shape)                   (n_agents*steps, 1)
        # print(np.array(predictions).shape)        (n_agents*steps, 50)
        # print(np.array(actions).shape)            (n_agents*steps, 50)

        world_map = []
        flat = []
        for s in states:
            world_map.append(
                tf.convert_to_tensor(
                    s["world-map"],
                )
            )

            flat.append(
                tf.convert_to_tensor(
                    s["flat"],
                )
            )

        y_true = tf.convert_to_tensor(y_true)
        world_map = tf.convert_to_tensor(world_map)
        flat = tf.convert_to_tensor(flat)

        logging.debug("     Fit Actor Network")

        # logging.debug(f"Epochs: {self.policy_config.agents_per_possible_policy}")
        # logging.debug(f"batch_size: {self.batch_size}")
        # logging.debug(f"worldmap size, flat size: {world_map.shape},{flat.shape}")

        # sys.exit()
        # training Actor and Critic networks
        a_loss = self.Actor.actor.fit(
            [world_map, flat],
            y_true,
            # batch_size=self.batch_size,
            epochs=self.policy_config.agents_per_possible_policy*self.policy_config.num_workers,
            steps_per_epoch=self.batch_size//self.policy_config.num_workers,
            verbose=0,
            shuffle=self.shuffle,
            # workers=8,
            # use_multiprocessing=True,
        )
        logging.debug(f"        Actor loss: {a_loss.history['loss'][-1]}")

        values = tf.convert_to_tensor(values)
        logging.debug("     Fit Critic Network")

        target = [target, values]

        c_loss = self.Critic.critic.fit(
            [world_map, flat],
            target,
            # batch_size=self.batch_size,
            epochs=1,
            steps_per_epoch=self.batch_size,
            verbose=0,
            shuffle=self.shuffle,
            # workers=8,
            # use_multiprocessing=True,
        )

        logging.debug(f"        Critic loss: {c_loss.history['loss'][-1]}")

    def _load(self) -> None:
        """
        Save Actor and Critic weights'
        """
        self.Actor.actor.load_weights(self.Actor_name)
        self.Critic.critic.load_weights(self.Critic_name)

    def _save(self) -> None:
        """
        Load Actor and Critic weights'
        """
        self.Actor.actor.save_weights(self.Actor_name)
        self.Critic.critic.save_weights(self.Critic_name)

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

    @deprecated
    def build_action_dict(self, obs: dict):
        """


        Build an action dictionary that can be used in training
        FIXME: right now, for developing reasons `p`'s policy doesn't exist and is not manged:
        so paller's action will be `0`

        Arguments:
            obs: environment observations

        Returns:
            A dictionary containing an action for each agent
        """
        actions = {}
        actions_oneshot = {}
        predictions = {}

        for key in obs.keys():
            if self._policy_mapping_fun(key) == "a":
                actions[key], actions_oneshot[key], predictions[key] = self.act(
                    obs[key]
                )
            elif self._policy_mapping_fun(key) == "p":
                actions["p"] = [0, 0, 0, 0, 0, 0, 0]
            else:
                IndexError(f"this actor is not managed by the environment, key: {key}")

        return actions, actions_oneshot, predictions

    @deprecated
    def train_one_step_with_batch(self, data):
        """
        Train agents for one step using mini_batching
        """
        data.batch
        self.learn()
