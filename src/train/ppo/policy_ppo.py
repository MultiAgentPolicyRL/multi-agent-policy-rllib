from typing import List, Tuple

import torch

from src.common import Model
from src.train.ppo import PytorchLinear#, RolloutBuffer
from src.train.ppo.utils.rollout_buffer import RolloutBuffer

class PpoPolicy(Model):
    """
    PPO Main Optimization Algorithm
    """
    # TODO: fix experiment_name and model saving/loading
    def __init__(
        self,
        observation_space,
        action_space,
        K_epochs: int = 16,
        eps_clip: int = 0.1,
        gamma: float = 0.998,
        c1: float = 0.5,
        c2: float = 0.01,
        learning_rate: float = 0.0003,
        device: str = "cpu",
        name: str = None # as an "a" or a "p"
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )
        # update policy for K epochs in one PPO update
        self.k_epochs = K_epochs
        # Clip parameter for PPO
        self.eps_clip = eps_clip
        # discount factor
        self.gamma = gamma
        # Hyperparameters in loss
        self._c1, self._c2 = c1, c2


        self.device = device
        # Environment and PPO parameters


        self.model: PytorchLinear = PytorchLinear(
            obs_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            learning_rate=learning_rate
        ).to(self.device)

        self.mse_loss = torch.nn.MSELoss()

        self.name = name

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
            policy_action, policy_probability = self.model.act(observation)

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
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.model.evaluate(
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

            critic_loss = self.mse_loss(state_values, rewards)

            # final loss of clipped objective PPO w/AI-Economist hyperparameters
            loss = (
                -torch.min(surr1, surr2)
                + self._c1 * critic_loss
                - self._c2 * dist_entropy
            )

            # take gradient step
            self.model.optimizer.zero_grad()
            loss.mean().backward()
            self.model.optimizer.step()

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
        actor_weights, critic_weights, optimizer_weights = self.model.get_weights()
        return {"a": actor_weights, "c": critic_weights, "o": optimizer_weights}

    def set_weights(self, weights: dict) -> None:
        """
        Set policy weights.
        """
        # print(f"updating weights")
        self.model.set_weights(
            actor_weights=weights["a"],
            critic_weights=weights["c"],
            optimizer_weights=weights["o"],
        )

    def save_model(self):
        """
        Save policy's model.
        """
        path = "saved_models/" + self.name + "_model.pt"
        torch.save(self.model, path)

    def load_model(self):
        """
        Load policy's model.
        """
        path = "saved_models/" + self.name + "_model.pt"
        self.model = torch.load(path)
