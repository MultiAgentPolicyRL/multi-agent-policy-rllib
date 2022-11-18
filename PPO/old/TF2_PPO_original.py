# from __future__ import absolute_import
import gym
import copy
import imageio
import datetime
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import tensorflow_probability as tfp

tfd = tfp.distributions

tf.keras.backend.set_floatx('float64')

# paper https://arxiv.org/pdf/1707.06347.pdf
# code references https://github.com/uidilr/ppo_tf/blob/master/ppo.py,
# https://github.com/openai/baselines/tree/master/baselines/ppo1
# https://github.com/anita-hu/TF2-RL/blob/master/PPO/TF2_PPO.py


def model(state_shape, action_dim, units=(400, 300, 100), discrete=True):
    state = Input(shape=state_shape)

    # Value function (baseline)
    # used to calculate advantage estimate
    vf = Dense(units[0], name="Value_L0", activation="tanh")(state)
    for index in range(1, len(units)):
        vf = Dense(units[index], name="Value_L{}".format(
            index), activation="tanh")(vf)

    value_pred = Dense(1, name="Out_value")(vf)

    # Our Policy
    pi = Dense(units[0], name="Policy_L0", activation="tanh")(state)
    for index in range(1, len(units)):
        pi = Dense(units[index], name="Policy_L{}".format(
            index), activation="tanh")(pi)

    if discrete:
        action_probs = Dense(action_dim, name="Out_probs",
                             activation='softmax')(pi)
        model = Model(inputs=state, outputs=[action_probs, value_pred])
    else:
        actions_mean = Dense(action_dim, name="Out_mean",
                             activation='tanh')(pi)
        model = Model(inputs=state, outputs=[actions_mean, value_pred])

    return model


class PPO:
    """
    PPO with gae, adam optimizer, auto-size neural network, for discrete environments.

    Parameters
    ----------
    env: gym
    discrete: bool
        Describes if the environment is discrete or continuous
    lr: float
        Learning Rate
    hidden_units: tuple
    c1: float
    c2: float
    clip_ratio: float
        Clip ratop for L_clip. Epsylon in PPO's paper
    gamma: float
        Discout factor
    lam: float
        Also known as lambda
    batch_size: int
    n_updates: int
    """

    def __init__(
            self,
            env,
            lr=5e-4,
            hidden_units=(24, 16),
            c1=1.0,
            c2=0.01,
            clip_ratio=0.2,
            gamma=0.95,
            lam=1.0,
            batch_size=64,
            n_updates=4,
    ):
        # Define environment, observation_shape and action_shape (or dimension)
        self.env = env
        self.state_shape = env.observation_space.shape  # shape of observations
        # number of actions
        self.action_dim = env.action_space.n 

        

        # Define and initialize network
        # Define and initialize Keras Model
        self.policy = model(self.state_shape, self.action_dim,
                            hidden_units, discrete=True)
        # Model optimizer used here is Adam, SGD in PPO's paper. (adam is faster)
        self.model_optimizer = Adam(learning_rate=lr)
        print(self.policy.summary())

        # Set hyperparameters
        self.gamma = gamma  # discount factor
        self.lam = lam
        self.c1 = c1  # value difference coeff
        self.c2 = c2  # entropy coeff
        self.clip_ratio = clip_ratio  # for clipped surrogate
        self.batch_size = batch_size
        self.n_updates = n_updates  # number of epochs per episode

        # Tensorboard
        self.summaries = {}

    def get_dist(self, output):
        """
        Categorical distribution
        ------------------------

            [From Wikipedia](https://en.wikipedia.org/wiki/Categorical_distribution)  \n
            In probability theory and statistics, a categorical distribution (also called a
            generalized Bernoulli distribution, multinoulli distribution[1]) is a discrete 
            probability distribution that describes the possible results of a random variable
            that can take on one of K possible categories, with the probability of each
            category separately specified. There is no innate underlying ordering of these
            outcomes, but numerical labels are often attached for convenience in describing
            the distribution, (e.g. 1 to K). The K-dimensional categorical distribution is
            the most general distribution over a K-way event; any other discrete distribution
            over a size-K sample space is a special case. The parameters specifying the
            probabilities of each possible outcome are constrained only by the fact that
            each must be in the range 0 to 1, and all must sum to 1.

        Parameters
        ----------
        output: action_mean

        Returns
        -------
        dist: Categorical Distribution
        """

        dist = tfd.Categorical(probs=output)
        return dist

    def evaluate_actions(self, obs, action):
        output, value = self.policy(obs)
        dist = self.get_dist(output)

        log_probs = dist.log_prob(action)
        entropy = dist.entropy()

        return log_probs, entropy, value

    def act(self, obs):
        obs = np.expand_dims(obs, axis=0).astype(np.float64)
        output, value = self.policy.predict(obs, verbose=0)
        dist = self.get_dist(output)

        action = dist.sample()
        log_probs = dist.log_prob(action)

        return action[0].numpy(), value[0][0], log_probs[0].numpy()

    def save_model(self, fn):
        """
        Save tf trained model
        """
        self.policy.save(fn)

    def load_model(self, fn):
        """
        Load tf trained model
        """
        self.policy.load_weights(fn)
        print(self.policy.summary())

    def get_gaes(self, rewards, v_preds, next_v_preds):
        """
            Calculate GAE - Truncated version of Generalized Advantage Estimation
        """

        # source: https://github.com/uidilr/ppo_tf/blob/master/ppo.py#L98
        deltas = [r_t + self.gamma * v_next - v for r_t,
                  v_next, v in zip(rewards, next_v_preds, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        # is T-1, where T is time step which run policy
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.lam * self.gamma * gaes[t + 1]
        return gaes

    def learn(self, observations, actions, log_probs, next_v_preds, rewards, gaes):

        rewards = np.expand_dims(rewards, axis=-1).astype(np.float64)
        next_v_preds = np.expand_dims(next_v_preds, axis=-1).astype(np.float64)

        with tf.GradientTape() as tape:
            new_log_probs, entropy, state_values = self.evaluate_actions(
                observations, actions)

            ratios = tf.exp(new_log_probs - log_probs)
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1-self.clip_ratio,
                                              clip_value_max=1+self.clip_ratio)
            loss_clip = tf.minimum(gaes * ratios, gaes * clipped_ratios)
            loss_clip = tf.reduce_mean(loss_clip)

            target_values = rewards + self.gamma * next_v_preds
            vf_loss = tf.reduce_mean(
                tf.math.square(state_values - target_values))

            entropy = tf.reduce_mean(entropy)
            total_loss = -loss_clip + self.c1 * vf_loss - self.c2 * entropy

        train_variables = self.policy.trainable_variables
        grad = tape.gradient(total_loss, train_variables)  # compute gradient
        self.model_optimizer.apply_gradients(zip(grad, train_variables))

        # tensorboard info
        self.summaries['total_loss'] = total_loss
        self.summaries['surr_loss'] = loss_clip
        self.summaries['vf_loss'] = vf_loss
        self.summaries['entropy'] = entropy

    def train(self, max_epochs=8000, max_steps=500, save_freq=50):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        episode, epoch = 0, 0

        while epoch < max_epochs:
            done, steps = False, 0
            cur_state = self.env.reset()
            obs, actions, log_probs, rewards, v_preds, next_v_preds = [], [], [], [], [], []

            # Collecting data for batching
            while not done and steps < max_steps:
                # determine action
                action, value, log_prob = self.act(cur_state)
                # act on env
                next_state, reward, done, infos = self.env.step(action)
                # self.env.render()

                rewards.append(reward)
                v_preds.append(value)
                obs.append(cur_state)
                actions.append(action)
                log_probs.append(log_prob)

                steps += 1
                cur_state = next_state

            next_v_preds = v_preds[1:] + [0]
            gaes = self.get_gaes(rewards, v_preds, next_v_preds)
            gaes = np.array(gaes).astype(dtype=np.float64)
            gaes = (gaes - gaes.mean()) / gaes.std()
            data = [obs, actions, log_probs, next_v_preds, rewards, gaes]

            for i in range(self.n_updates):
                # Sample training data
                sample_indices = np.random.randint(
                    low=0, high=len(rewards), size=self.batch_size)
                sampled_data = [
                    np.take(a=a, indices=sample_indices, axis=0) for a in data]

                # Train model
                self.learn(*sampled_data)

                # Tensorboard update
                with summary_writer.as_default():
                    tf.summary.scalar('Loss/total_loss',
                                      self.summaries['total_loss'], step=epoch)
                    tf.summary.scalar('Loss/clipped_surr',
                                      self.summaries['surr_loss'], step=epoch)
                    tf.summary.scalar(
                        'Loss/vf_loss', self.summaries['vf_loss'], step=epoch)
                    tf.summary.scalar(
                        'Loss/entropy', self.summaries['entropy'], step=epoch)

                summary_writer.flush()
                epoch += 1

            episode += 1
            print("episode {}: {} total reward, {} steps, {} epochs".format(
                episode, np.sum(rewards), steps, epoch))

            # Tensorboard update
            with summary_writer.as_default():
                tf.summary.scalar('Main/episode_reward',
                                  np.sum(rewards), step=episode)
                tf.summary.scalar('Main/episode_steps', steps, step=episode)

            summary_writer.flush()

            if steps >= max_steps:
                print("episode {}, reached max steps".format(episode))
                self.save_model("ppo_episode{}.h5".format(episode))

            if episode % save_freq == 0:
                self.save_model("ppo_episode{}.h5".format(episode))

        self.save_model("ppo_final_episode{}.h5".format(episode))

    def test(self, render=True, fps=30, filename='test_render.mp4'):
        cur_state, done, rewards = self.env.reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            action, value, log_prob = self.act(cur_state, test=True)
            next_state, reward, done, _ = self.env.step(action)
            cur_state = next_state
            rewards += reward
            if render:
                video.append_data(self.env.render(mode='rgb_array'))
        video.close()
        return rewards


if __name__ == "__main__":
    gym_env = gym.make("CartPole-v1")
    # gym_env = gym.make("Pendulum-v0")
    
    ppo = PPO(gym_env)

    # ppo.load_model("basic_models/ppo_episode176.h5")
    ppo.train(max_epochs=1000, save_freq=50)
    # reward = ppo.test()
    # print("Total rewards: ", reward)
