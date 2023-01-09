"""
Rollout worker. This good guy manages its policy and creates a batch
"""
from policies import Policy, PpoPolicy, EmptyPolicy
from typing import Dict, Tuple

from utils.rollout_buffer import Memory
from utils.execution_time import exec_time

def build_policy(policy_config):
    """
    Builds a policy from its config.

    To add a policy to this registry just add it to the list.
    """
    if policy_config['policy'] == EmptyPolicy:
        return EmptyPolicy(
            observation_space=policy_config['observation_space'],
            action_space=policy_config['action_space'])
    elif policy_config['policy'] == PpoPolicy:
        return PpoPolicy(
            observation_space=policy_config['observation_space'],
            action_space=policy_config['action_space'],
            K_epochs=policy_config['K_epochs'],
            device=policy_config['device']
        )
    else:
        KeyError(f"Policy {policy_config['policy']} is not in the registry")

class RolloutWorker:
    """
    Rollout worker. It manages its policy and creates a batch.
    At the moment it doesn't return anything -> need to add a return system to get
    Env mean reward per batch

    Args:
        rollout_fragment_length
    """

    def __init__(
        self,
        rollout_fragment_length: int,
        policies_config: dict,
        available_agent_id: list,
        policies_size: dict,
        policy_mapping_function,
        env,
        device: str,
    ):
        self.env = env
        # self.policies_config = policies_config
        self.rollout_fragment_length = rollout_fragment_length
        self.policies_size = policies_size
        self.available_agent_id = available_agent_id
        self.policy_mapping_function = policy_mapping_function

        self.policies = {}
        for key in policies_config.keys():
            self.policies[key] = build_policy(policies_config[key])

        self.memory = Memory(
            policy_mapping_fun=self.policy_mapping_function,
            available_agent_id=self.available_agent_id,
            batch_size=self.rollout_fragment_length,
            policy_size=self.policies_size,
            device=device,
        )

    @exec_time
    def batch(self):
        """
        Creates a batch of `rollout_fragment_length` steps, save in `self.rollout_buffer`.
        """
        # reset batching environment and get its observation
        obs = self.env.reset()
        # reset rollout_buffer
        self.memory.clear()

        for counter in range(self.rollout_fragment_length):
            # get actions, action_logprob for all agents in each policy* wrt observation
            policy_action, policy_logprob = self.get_actions(obs)
            # get new_observation, reward, done from stepping the environment
            next_obs, rew, done, _ = self.env.step(policy_action)

            # save new_observation, reward, done, action, action_logprob in rollout_buffer
            self.memory.update(
                state=obs,
                action=policy_action,
                logprob=policy_logprob,
                reward=rew,
                is_terminal=done,
            )

            if counter == self.rollout_fragment_length - 1:
                # end this episode, start a new one. Set done to True (used in policies)
                # reset batching environment and get its observation
                # next_obs = self.env.reset()
                done['__all__'] = True
                self.memory.update(
                    state=obs,
                    action=policy_action,
                    logprob=policy_logprob,
                    reward=rew,
                    is_terminal=done,
                )
                break

            obs = next_obs

    # @exec_time
    def get_actions(self, obs: dict) -> Tuple[dict, dict]:
        """
        Build action dictionary using actions taken from all policies.

        Args:
            obs: environment observation

        Returns:
            policy_action
            policy_logprob
        """

        policy_action, policy_logprob = {}, {}

        for key in obs.keys():
            (policy_action[key], policy_logprob[key]
             ) = self.policies[self.policy_mapping_function(key)].act(obs[key])

        return policy_action, policy_logprob

    def learn(self, memory):
        """
        TODO: docs
        """
        for key in self.policies:
            self.policies[key].learn(
            rollout_buffer=memory.get(key))

    def get_weights(self) -> dict:
        """
        Get model weights
        """
        weights = {}
        for key in self.policies.keys():
            weights[key] = self.policies[key].get_weights()

        return weights

    def set_weights(self, weights: dict):
        """
        Set model weights
        """
        for key in self.policies.keys():
            self.policies[key].set_weights(weights[key])
