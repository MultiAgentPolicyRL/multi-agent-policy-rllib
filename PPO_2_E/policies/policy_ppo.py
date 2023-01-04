import random
import sys
from typing import List, Tuple
import torch
from models import PytorchLinear
from policies import Policy
from utils import RolloutBuffer
from utils import exec_time


class PpoPolicy(Policy):
    """
    PPO Main Optimization Algorithm
    """

    def __init__(self, observation_space, action_space, K_epochs, device):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )

        ## TMP: parameters
        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.001  # learning rate for critic network

        self.K_epochs = K_epochs     # update policy for K epochs in one PPO update
        self.eps_clip = 0.2  # clip parameter for PPO
        self.gamma = 0.99  # discount factor
        self.device = device
        # Environment and PPO parameters
        self.Model: PytorchLinear = PytorchLinear(
            obs_space=self.observation_space,
            action_space=self.action_space,
            device=self.device
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.Model.actor.parameters(), "lr": lr_actor},
                {"params": self.Model.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.MseLoss = torch.nn.MSELoss()

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
        # Get the prediction from the Actor network
        with torch.no_grad():
            policy_action, policy_probability = self.Model.act(observation)

        return policy_action.item(), policy_probability

    @exec_time
    def learn(
        self,
        rollout_buffer: List[RolloutBuffer],
    ) -> Tuple[float, float]:
        """
        Train Policy networks
        Takes as input the batch with N epochs of M steps_per_epoch. As we are using an LSTM
        model we are not shuffling all the data to create the minibatch, but only shuffling
        each epoch.

        Example:
            Input epochs: 0,1,2,3
            Shuffled epochs: 2,0,1,3

        It calls `self.Model.fit` passing the shuffled epoch.

        Args:
            rollout_buffer: RolloutBuffer for this specific policy.
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
        # FIXME: make it work with a list of RolloutBuffers or a single RolloutBuffer
        epochs_order = list(range(rollout_buffer[0].n_agents))
        steps_per_epoch = rollout_buffer[0].batch_size
        random.shuffle(epochs_order)


        a_loss, c_loss = [], []
        for i in epochs_order:
            a, c = self.__update(rollout_buffer[i])

            a_loss.append(a)
            c_loss.append(c)

        a_loss = torch.mean(torch.tensor(a_loss))
        c_loss = torch.mean(torch.tensor(c_loss))

        return a_loss.float(), c_loss.float()

    def __update(self, buffer: RolloutBuffer):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(
            reversed(buffer.rewards), reversed(buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = buffer.states
        old_actions= buffer.actions
        old_logprobs = buffer.logprobs

        a_loss, c_loss = [], []

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.Model.evaluate(
                old_states, old_actions
            )
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio pi_theta / pi_theta_old
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            critic_loss = self.MseLoss(state_values, rewards)
            # final loss of clipped objective PPO

            loss = -torch.min(surr1, surr2) + 0.5 * critic_loss - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            c_loss.append(torch.mean(loss))
            a_loss.append(torch.mean(loss))

        a_loss = torch.mean(torch.tensor(a_loss))
        c_loss = torch.mean(torch.tensor(c_loss))

        return a_loss, c_loss
