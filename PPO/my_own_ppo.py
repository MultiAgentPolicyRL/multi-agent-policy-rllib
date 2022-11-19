"""
docs about this file
"""
from ast import Tuple
import copy
import random
import sys
import keras as k
import tensorflow as tf
import numpy as np

from tensorflow.python.framework.ops import enable_eager_execution

enable_eager_execution()

def dict_to_tensor_dict(a_dict: dict):
    """
    pass a single agent obs, returns it's tensor_dict
    """
    tensor_dict = {}
    for key, value in a_dict.items():
        tensor_dict[key] = tf.convert_to_tensor(value, name=key)
        tensor_dict[key] = tf.expand_dims(tensor_dict[key], axis=0)

    return tensor_dict


class ActorModel(object):
    """
    a
    """

    def __init__(self, action_space: int = 6) -> k.Model:
        """
        Builds the model. Takes in input the parameters that were not specified in the paper.
        """
        self.action_space = action_space

        self.cnn_in = k.Input(shape=(7, 11, 11))
        self.map_cnn = k.layers.Conv2D(16, 3, activation="relu")(self.cnn_in)
        self.map_cnn = k.layers.Conv2D(32, 3, activation="relu")(self.map_cnn)
        self.map_cnn = k.layers.Flatten()(self.map_cnn)

        self.info_input = k.Input(shape=(136))
        self.mlp1 = k.layers.Concatenate()([self.map_cnn, self.info_input])
        self.mlp1 = k.layers.Dense(128, activation="relu")(self.mlp1)
        self.mlp1 = k.layers.Dense(128, activation="relu")(self.mlp1)
        self.mlp1 = k.layers.Reshape([1, -1])(self.mlp1)

        self.lstm = k.layers.LSTM(128)(self.mlp1)

        # Policy pi - needs to be a probabiliy value
        self.action_probs = k.layers.Dense(
            action_space, name="Out_probs_actions", activation="sigmoid"
        )(self.lstm)

        self.actor : k.Model = k.Model(
            inputs=[self.cnn_in, self.info_input], outputs=self.action_probs
        )

        # reason of Adam optimizer lr=0.0003 https://github.com/ray-project/ray/issues/8091
        self.actor.compile(
            optimizer=k.optimizers.Adam(lr=0.0003), loss=self.ppo_loss
        )

    def ppo_loss(self, y_true, y_pred):
        """
        Defined in https://arxiv.org/abs/1707.06347
        """
        advantages, prediction_picks, actions = (
            y_true[:, :1],
            y_true[:, 1 : 1 + self.action_space],
            y_true[:, 1 + self.action_space :],
        )
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = k.backend.clip(prob, 1e-10, 1.0)
        old_prob = k.backend.clip(old_prob, 1e-10, 1.0)

        ratio = k.backend.exp(k.backend.log(prob) - k.backend.log(old_prob))

        p1 = ratio * advantages
        p2 = (
            k.backend.clip(
                ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING
            )
            * advantages
        )

        actor_loss = -k.backend.mean(k.backend.minimum(p1, p2))

        entropy = -(y_pred * k.backend.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * k.backend.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, obs):
        """
        If you remove the reshape good luck finding that softmax sum != 1.
        """
        obs = dict_to_tensor_dict(obs)
        prediction = tf.reshape(
            self.actor.predict([obs["world-map"], obs["flat"]], verbose=False), [-1]
        )

        # return self.actor.predict(state)
        return prediction / np.sum(prediction)


class CriticModel(object):
    """
    a
    """

    def __init__(self) -> k.Model:
        """Builds the model. Takes in input the parameters that were not specified in the paper."""
        old_values = k.Input(shape=(1,))
        cnn_in = k.Input(shape=(7, 11, 11))
        map_cnn = k.layers.Conv2D(16, 3, activation="relu")(cnn_in)
        map_cnn = k.layers.Conv2D(32, 3, activation="relu")(map_cnn)
        map_cnn = k.layers.Flatten()(map_cnn)

        info_input = k.Input(shape=(136))
        mlp1 = k.layers.Concatenate()([map_cnn, info_input])
        mlp1 = k.layers.Dense(128, activation="relu")(mlp1)
        mlp1 = k.layers.Dense(128, activation="relu")(mlp1)
        mlp1 = k.layers.Reshape([1, -1])(mlp1)

        lstm = k.layers.LSTM(128)(mlp1)

        value_pred = k.layers.Dense(
            1, name="Out_value_function", activation="softmax"
        )(lstm)

        self.critic : k.Model = k.Model(
            inputs=[cnn_in, info_input, old_values], outputs=value_pred
        )

        # reason of Adam optimizer https://github.com/ray-project/ray/issues/8091
        # 0.0003
        self.critic.compile(
            optimizer=k.optimizers.Adam(lr=0.0003), loss=self.critic_ppo2_loss(old_values)
        )

    def critic_ppo2_loss(self, values):
        """
        returns loss function
        """

        def loss(y_true, y_pred):
            """
            PPO's loss function, can be with mean or clipped
            """
            # standard PPO loss
            # value_loss = k.backend.mean((y_true - y_pred) ** 2)

            # L_CLIP
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + k.backend.clip(
                y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING
            )
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            value_loss = 0.5 * k.backend.mean(k.backend.maximum(v_loss1, v_loss2))
            return value_loss

        return loss

    def predict(self, obs_predict):
        """
        a
        """
        # obs_predict = dict_to_tensor_dict(obs_predict)
        # from the guy's code
        # return self.critic.predict([obs, np.zeros((obs.shape[0], 1))])
        # return self.critic.predict([obs_predict["world-map"], obs_predict["flat"], np.zeros((7, 136))], verbose=False)
        return self.critic.predict([k.backend.expand_dims(obs_predict["world-map"], 0),k.backend.expand_dims(obs_predict["flat"], 0),k.backend.expand_dims(np.zeros((136, 1)), 0),])

class PPOAgent:
    """
    PPO Main Optimization Algorithm
    """

    def __init__(self, env_name ):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name
        # self.env = env
        self.action_size = 6  # self.env.action_space.n
        # self.state_size = self.env.observation_space.shape
        self.EPISODES = 10000  # total episodes to train through all environments
        self.episode = 0  # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0  # when average score is above 0 model will be saved
        # self.lr = 0.00025
        self.epochs = 10  # training epochs
        self.shuffle = False
        self.mini_batching_steps = 10 # mini-batching
        # self.optimizer = RMSprop
        # self.optimizer = Adam

        self.replay_count = 0
        # self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))

        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = (
            [],
            [],
            [],
        )  # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = ActorModel(self.action_size)
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
        action = random.choices(np.arange(self.action_size), weights=prediction)[0]
        # action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
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
        # FIXME
        Removed dones
        """
        deltas = [
            r + gamma * nv - v
            for r, nv, v in zip(rewards, next_values, values)
        ]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)

        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + gamma * lamda * gaes[t + 1]

        target = gaes + values

        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return np.vstack(gaes), np.vstack(target)

    def _replay(
        self, observations: dict, actions: dict, rewards: dict, predictions: dict, dones: dict, next_observations: dict
    ):
        """
        Train Critic network

        FIXME: Arrived here, need to understand this func better.
        """
        # For each "timestep" (0,1,2,3,p)
        for obs_time, act_time, rew_time, pred_time, next_obs_time in zip(observations.values(), actions.values(), rewards.values(), predictions.values(), next_observations.values()):
            for key in obs_time.keys():
                # print(obs_time[key])
                if self._policy_mapping_fun(key) == 'a':
                    print("pasta")
                    observations = obs_time[key]
                    next_observations= next_obs_time[key]
                    rewards= rew_time[key]
                    predictions = pred_time[key]
                    actions = act_time[key]
                    # observations, actions, rewards, predictions, dones, next_observations = observations1, actions1, rewards1, predictions1, dones1, next_observations1

                    # reshape memory to appropriate shape for training
                    # observations = np.vstack(observations)
                    # next_observations = np.vstack(next_observations)
                    # actions = np.vstack(actions)
                    # predictions = np.vstack(predictions)

                    
                    # Get Critic network predictions
                    values = self.Critic.predict(observations)
                    next_values = self.Critic.predict(next_observations)

                    # Compute discounted rewards and advantages
                    advantages, target = self._get_gaes(
                        rewards, dones, np.squeeze(values), np.squeeze(next_values)
                    )

                    # stack everything to numpy array
                    # pack all advantages, predictions and actions to y_true and when they are received
                    # in custom PPO loss function we unpack it
                    y_true = np.hstack([advantages, predictions, actions])

                    # training Actor and Critic networks, return a_loss, c_loss
                    self.Actor.actor.fit(
                        observations, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle
                    )

                    self.Critic.critic.fit(
                        [observations, actions],
                        target,
                        epochs=self.epochs,
                        verbose=0,
                        shuffle=self.shuffle,
                    )

                self.replay_count += 1

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
            if self._policy_mapping_fun(key) == 'a':
                actions[key], actions_oneshot[key], predictions[key] = self.act(obs[key])
            elif self._policy_mapping_fun(key) == 'p':
                actions['p'] = [0,0,0,0,0,0,0]
            else:
                IndexError(f"this actor is not managed by the environment, key: {key}")

        return actions, actions_oneshot, predictions


    def train_one_step_batching(self, env):  # train every self.Training_batch episodes
        """
        Train agents for one step using mini_batching
        """
        
        state = env.reset()
        done, score = False, 0

        # Instantiate or reset games memory
        states, next_states, actions, rewards, predictions, dones = ({},{},{},{},{},{})
        for t in range(self.mini_batching_steps):
            # Actor picks an action
            # action, action_onehot, prediction = self.act(state)
            action, action_onehot, prediction = self.build_action_dict(state)

            # Retrieve new state, reward, and whether the state is terminal
            # print(action)
            next_state, reward, done, _ = env.step(action)
            # Memorize (state, action, reward) for training
            states[t]=state
            next_states[t]=next_state
            actions[t]=action_onehot
            rewards[t]=reward
            dones[t] = done
            predictions[t]=prediction
            
            # print(f"reward during batching: {reward}")

            # Update current state
            state = next_state
            
        self._replay(states, actions, rewards, predictions, dones, next_states)

if __name__ == "__main__":
    from main_test import get_environment
    env = get_environment()
    ppo_policy = PPOAgent("abla")
    ppo_policy.train_one_step_batching(env)
    
