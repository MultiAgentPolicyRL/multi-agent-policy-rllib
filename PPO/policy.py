"""
docs about this file
"""
import copy
import random
import sys
import numpy as np
from tensorflow.python.framework.ops import enable_eager_execution
from model import ActorModel, CriticModel
from deprecated import deprecated
import tensorflow as tf

# from algorithm import BatchMemory
enable_eager_execution()


class PPOAgent:
    """
    PPO Main Optimization Algorithm
    """

    def __init__(self, env_name="default", policy_config=None):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name
        self.action_space = 6  # self.env.action_space.n
        # self.state_size = self.env.observation_space.shape
        self.EPISODES = 10000  # total episodes to train through all environments
        self.episode = 0  # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0  # when average score is above 0 model will be saved
        self.epochs = 10  # training epochs
        self.shuffle = False
        self.mini_batching_steps = 10  # mini-batching

        if policy_config is not None:
            # NotImplementedError("Policy config injection has not been implemented yet.")
            self.action_space = policy_config['action_space']
            self.observation_space = policy_config['observation_space']

        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = (
            [],
            [],
            [],
        )  # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = ActorModel(self.action_space)
        self.Critic = CriticModel()

        self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.h5"

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
        prediction = self.Actor.predict(state)
        action = int(random.choices(state['action_mask'], weights=prediction)[0])
        # action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_space])
        action_onehot[action] = 1
        return action, action_onehot, prediction

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

        return np.vstack(gaes), np.vstack(target)

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
        # states = np.vstack(states)
        # next_states = np.vstack(next_states)
        # actions = np.vstack(actions)
        # predictions = np.vstack(predictions)

        # Get Critic network predictions
        values = self.Critic.batch_predict(states)
        next_values = self.Critic.batch_predict(next_states)

        # Compute discounted rewards and advantages
        # GAE
        advantages, target = self._get_gaes(
            rewards, np.squeeze(values), np.squeeze(next_values)
        )

        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        y_true = np.hstack([advantages, predictions, actions])

        # ['world-map', 'world-idx_map', 'time', 'flat', 'action_mask'])
        # tf.convert_to_tensor(
        #     value, dtype=None, dtype_hint=None, name=None
        # )

        world_map = [] 
        flat = []
        for s in states:
            world_map.append(tf.convert_to_tensor(s['world-map'],))
            flat.append(tf.convert_to_tensor(s['flat'],))

        world_map = tf.convert_to_tensor(world_map)
        flat = tf.convert_to_tensor(flat)



        # training Actor and Critic networks
        a_loss = self.Actor.actor.fit(
            # states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle
            [world_map, flat], y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle

        )

        c_loss = self.Critic.critic.fit(
            [states, values],
            target,
            epochs=self.epochs,
            verbose=0,
            shuffle=self.shuffle,
        )

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
