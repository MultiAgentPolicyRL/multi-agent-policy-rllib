"""
docs about this file
"""
import copy
import logging
import random
import sys
import time
from functools import wraps

import numpy as np
import tensorflow as tf
import torch 
from deprecated import deprecated
from model.model import LSTMModel
from policy.policy_config import PolicyConfig


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
        self.max_average = 0  # when average score is above 0 model will be saved
        self.batch_size = self.policy_config.batch_size  # training epochs
        self.shuffle = False

        # # if policy_config is not None:
        # #     self.action_space = policy_config["action_space"]
        # #     self.observation_space:  gym.spaces = policy_config["observation_space"]

        # # Instantiate plot memory
        # self.scores_, self.episodes_, self.average_ = (
        #     [],
        #     [],
        #     [],
        # )  # used in matplotlib plots

        # # Create Actor-Critic network models
        # self.Actor = ActorModel(policy_config.model_config)
        # self.Critic = CriticModel(policy_config.model_config)

        self.Model : LSTMModel = LSTMModel(obs=policy_config.observation_space, name="Model_a")

        # self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        # self.Critic_name = f"{self.env_name}_PPO_Critic.h5"

    def act(self, state):
        """
        Gets an action and vf value from the model.

        Args:
            state: actor state (dict)
        
        Returns:
            action: single action in [0, self.action_space] from logits distribution
            action_one_hot: one hot encoding for selected action 
            logits: actions probabiliy distribution
            value: vf value

        """
        # Use the network to predict the next action to take, using the model
        # Logits: action distribution w/applied mask 
        # Value: value function result
        # for key in state.keys():
        #     state[key] = torch.tensor(state[key])

        input_state = [
                torch.FloatTensor(state["world-map"]).unsqueeze(0),
                torch.FloatTensor(state["world-idx_map"]).unsqueeze(0),
                torch.FloatTensor(state["time"]).unsqueeze(0),
                torch.FloatTensor(state["flat"]).unsqueeze(0),
                torch.FloatTensor(state["action_mask"]).unsqueeze(0),
            ]

        # Get the prediction from the Actor network        
        with torch.no_grad():
            logits, value = self.Model(input_state)

        prediction = torch.squeeze(logits)
        # print(value)
        # Sample an action from the prediction distribution
        action = torch.FloatTensor(random.choices(np.arange(self.action_space), weights=prediction.detach().numpy()))
        
        # One-hot encode the action
        action_onehot = torch.zeros([self.action_space])
        action_onehot[int(action.item())] = 1

        return action, action_onehot, logits, value

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
        observations: list,
        next_observations: list,
        policy_actions: list,
        policy_predictions: list,
        rewards: list,
        vf_predictions: list,
        vf_predictions_old: list,
    ):
        """
        Train Policy networks
        """
        # Get Critic network predictions
        tempo = time.time()
        sys.exit()

        values = self.Critic.batch_predict(observations)
        next_values = self.Critic.batch_predict(next_observations)
        
        logging.debug(f"     Values and next_values required {time.time()-tempo}s")

        # Compute discounted rewards and advantages
        # GAE
        tempo = time.time()
        
        advantages, target = self._get_gaes(
            rewards, np.squeeze(values), np.squeeze(next_values)
        )

        logging.debug(f"     Gaes required {time.time()-tempo}s")
        
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        tempo = time.time()
        y_true = np.hstack([advantages, vf_predictions, policy_actions])
        logging.debug(f"     Data prep required: {time.time()-tempo}s")

        tempo = time.time()

        # training Actor and Critic networks
        a_loss = self.Actor.actor.fit(
            [world_map, flat],
            y_true,
            # batch_size=self.batch_size,
            epochs=self.policy_config.agents_per_possible_policy*self.policy_config.num_workers,
            steps_per_epoch=self.batch_size//self.policy_config.num_workers,
            verbose=0,
            shuffle=self.shuffle,
            workers=8,
            use_multiprocessing=True,
        )
        logging.debug(f"     Fit Actor Network required {time.time()-tempo}s")
        logging.debug(f"        Actor loss: {a_loss.history['loss'][-1]}")

        tempo = time.time()
        values = tf.convert_to_tensor(values)
        target = [target, values]
        logging.debug(f"    Prep 2 required {time.time()-tempo}")

        tempo = time.time()
        c_loss = self.Critic.critic.fit(
            [world_map, flat],
            target,
            # batch_size=self.batch_size,
            epochs=1,
            steps_per_epoch=self.batch_size,
            verbose=0,
            shuffle=self.shuffle,
            workers=8,
            use_multiprocessing=True,
        )
        logging.debug(f"     Fit Critic Network required {time.time()-tempo}s")

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
