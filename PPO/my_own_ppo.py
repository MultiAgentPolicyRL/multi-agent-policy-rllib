"""
docs about this file
"""
import copy
import sys
import keras as k
import tensorflow as tf
import numpy as np
from main_test import get_environment, dict_to_tensor_dict


class ActorModel(object):
    """
    a
    """

    def __init__(self, action_space: int = 6) -> k.Model:
        """ 
        Builds the model. Takes in input the parameters that were not specified in the paper.
        """
        self.action_space = action_space

        self.cnn_in = tf.keras.Input(shape=(7, 11, 11))
        self.map_cnn = tf.keras.layers.Conv2D(16, 3, activation='relu')(self.cnn_in)
        self.map_cnn = tf.keras.layers.Conv2D(32, 3, activation='relu')(self.map_cnn)
        self.map_cnn = tf.keras.layers.Flatten()(self.map_cnn)

        self.info_input = tf.keras.Input(shape=(136))
        self.mlp1 = tf.keras.layers.Concatenate()([self.map_cnn, self.info_input])
        self.mlp1 = tf.keras.layers.Dense(128, activation='relu')(self.mlp1)
        self.mlp1 = tf.keras.layers.Dense(128, activation='relu')(self.mlp1)
        self.mlp1 = tf.keras.layers.Reshape([1, -1])(self.mlp1)

        self.lstm = tf.keras.layers.LSTM(128)(self.mlp1)

        # Policy pi - needs to be a probabiliy value
        self.action_probs = tf.keras.layers.Dense(action_space, name="Out_probs_actions", activation='softmax')(self.lstm)

        self.actor = tf.keras.Model(inputs=[self.cnn_in, self.info_input], outputs=self.action_probs)

        # reason of Adam optimizer lr=0.0003 https://github.com/ray-project/ray/issues/8091
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003), loss=self.ppo_loss)

    def ppo_loss(self, y_true, y_pred):
        """
            Defined in https://arxiv.org/abs/1707.06347
        """
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = k.backend.clip(prob, 1e-10, 1.0)
        old_prob = k.backend.clip(old_prob, 1e-10, 1.0)

        ratio = k.backend.exp(k.backend.log(prob) - k.backend.log(old_prob))

        p1 = ratio * advantages
        p2 = k.backend.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -k.backend.mean(k.backend.minimum(p1, p2))

        entropy = -(y_pred * k.backend.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * k.backend.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, obs):
        """
        If you remove the reshape good luck finding that softmax sum != 1.
        """
        prediction = tf.reshape(self.actor.predict([obs['world-map'], obs['flat']], verbose = False), [-1])
        
        # return self.actor.predict(state)
        return prediction / np.sum(prediction)


class CriticModel(object):
    """
    a
    """

    def __init__(self) -> k.Model:
        """ Builds the model. Takes in input the parameters that were not specified in the paper. """
        self.cnn_in = tf.keras.Input(shape=(7, 11, 11))
        self.map_cnn = tf.keras.layers.Conv2D(16, 3, activation='relu')(self.cnn_in)
        self.map_cnn = tf.keras.layers.Conv2D(32, 3, activation='relu')(self.map_cnn)
        self.map_cnn = tf.keras.layers.Flatten()(self.map_cnn)

        self.info_input = tf.keras.Input(shape=(136))
        self.mlp1 = tf.keras.layers.Concatenate()([self.map_cnn, self.info_input])
        self.mlp1 = tf.keras.layers.Dense(128, activation='relu')(self.mlp1)
        self.mlp1 = tf.keras.layers.Dense(128, activation='relu')(self.mlp1)
        self.mlp1 = tf.keras.layers.Reshape([1, -1])(self.mlp1)

        self.lstm = tf.keras.layers.LSTM(128)(self.mlp1)

        self.value_pred = tf.keras.layers.Dense(1, name="Out_value_function", activation='softmax')(self.lstm)

        self.critic = tf.keras.Model(inputs=[self.cnn_in, self.info_input], outputs=self.value_pred)

        # reason of Adam optimizer https://github.com/ray-project/ray/issues/8091
        # 0.0003
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0003), loss=self.critic_ppo2_loss)

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
            clipped_value_loss = values + k.backend.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            value_loss = 0.5 * k.backend.mean(k.backend.maximum(v_loss1, v_loss2))
            return value_loss
        return loss

    def predict(self, obs):
        """
        a
        """
        # from the guy's code
        # return self.critic.predict([obs, np.zeros((obs.shape[0], 1))])
        return self.critic.predict([obs['world-map'], obs['flat']], verbose=False)
        

class PPOAgent:
    """
        PPO Main Optimization Algorithm
    """

    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name
        self.env = get_environment()
        self.action_size = 6 # self.env.action_space.n
        # self.state_size = self.env.observation_space.shape
        self.EPISODES = 10000  # total episodes to train through all environments
        self.episode = 0  # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0  # when average score is above 0 model will be saved
        # self.lr = 0.00025
        self.epochs = 10  # training epochs
        self.shuffle = False
        self.Training_batch = 1000
        #self.optimizer = RMSprop
        # self.optimizer = Adam

        self.replay_count = 0
        # self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))

        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], []  # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = ActorModel(self.action_size)
        self.Critic = CriticModel()

        self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.h5"

    def act(self, state):
        """ example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.998, lamda=0.98, normalize=True):
        """
        Gae's calculation
        Untouched, as it should be
        """
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, observations, actions, rewards, predictions, dones, next_observations):
        """
        FIXME: Arrived here, need to understand this func better.
        """
        # reshape memory to appropriate shape for training
        observations = np.vstack(observations)
        next_observations = np.vstack(next_observations)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions
        values = self.Critic.predict(observations)
        next_values = self.Critic.predict(next_observations)

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        y_true = np.hstack([advantages, predictions, actions])

        # training Actor and Critic networks
        a_loss = self.Actor.actor.fit(observations, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.critic.fit([observations, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        self.replay_count += 1

        
if __name__ == "__main__":
    ppo_agent = PPOAgent("abla")
    obs = ppo_agent.env.reset()
    print(ppo_agent.act(dict_to_tensor_dict(obs['0'])))