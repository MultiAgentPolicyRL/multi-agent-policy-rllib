import random
import sys
from typing import List, Tuple
import torch
from trainer.models import PytorchLinear, LSTMModel
from trainer.policies import Policy
from trainer.utils import RolloutBuffer, exec_time
from tensordict import TensorDict


class PpoPolicy(Policy):
    """
    PPO Main Optimization Algorithm
    """

    def __init__(self, observation_space, action_space, K_epochs, device):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )
        self.K_epochs = K_epochs  # update policy for K epochs in one PPO update
        self.eps_clip = 10  # 0.2 # clip parameter for PPO
        self.gamma = 0.998  # discount factor
        self.device = device
        # Environment and PPO parameters
        self.Model: PytorchLinear = PytorchLinear(
            obs_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
        ).to(self.device)

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

    # @exec_time
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
        buffer = rollout_buffer.to_tensor()
        a_loss, c_loss = self.__update(buffer=buffer)
        return a_loss, c_loss

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
        old_actions = buffer.actions
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

            # final loss of clipped objective PPO w/AI-Economist hyperparameters
            loss = -torch.min(surr1, surr2) + 0.05 * critic_loss - 0.025 * dist_entropy

            # My original loss
            # loss = -torch.min(surr1, surr2) + 0.5 * critic_loss - 0.01 * dist_entropy

            # take gradient step
            self.Model.optimizer.zero_grad()
            loss.mean().backward()
            self.Model.optimizer.step()

            c_loss.append(torch.mean(critic_loss))
            a_loss.append(torch.mean(loss))

        a_loss = torch.mean(torch.tensor(a_loss)).numpy()
        c_loss = torch.mean(torch.tensor(c_loss)).numpy()

        return a_loss, c_loss

    def get_weights(self) -> dict:
        """
        Get policy weights.

        Return:
            actor_weights, critic_weights
        """
        actor_weights, critic_weights, optimizer_weights = self.Model.get_weights()
        return {"a": actor_weights, "c": critic_weights, "o": optimizer_weights}

    def set_weights(self, weights: dict) -> None:
        """
        Set policy weights.
        """
        # FIXME: fix input
        # FIXME: add args
        # print(f"updating weights")
        self.Model.set_weights(
            actor_weights=weights["a"],
            critic_weights=weights["c"],
            optimizer_weights=weights["o"],
        )
