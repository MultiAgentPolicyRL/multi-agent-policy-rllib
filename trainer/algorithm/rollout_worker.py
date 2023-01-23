"""
Rollout worker. This good guy manages its policy and creates a batch
"""
# import dill as pickle
import pickle
from typing import Dict, Tuple

from trainer.policies import EmptyPolicy, PpoPolicy
from trainer.utils.execution_time import exec_time
from trainer.utils.rollout_buffer import RolloutBuffer


def build_policy(policy_config):
    """
    Builds a policy from its config.

    To add a policy to this registry just add it to the list.
    """
    if policy_config["policy"] == EmptyPolicy:
        return EmptyPolicy(
            observation_space=policy_config["observation_space"],
            action_space=policy_config["action_space"],
        )
    elif policy_config["policy"] == PpoPolicy:
        return PpoPolicy(
            observation_space=policy_config["observation_space"],
            action_space=policy_config["action_space"],
            K_epochs=policy_config["K_epochs"],
            device=policy_config["device"],
        )
    else:
        KeyError(f"Policy {policy_config['policy']} is not in the registry")


class RolloutWorker:
    """
    Rollout worker. It manages its policy and creates a batch.
    At the moment it doesn't return anything -> need to add a return system to get
    Env mean reward per batch

    Args:
        rollout_fragment_length: int,
        policies_config: dict,
        available_agent_id: list,
        policies_size: dict,
        policy_mapping_function,
        env,
        device: str,
        id,
        experiment_name,
    """

    def __init__(
        self,
        rollout_fragment_length: int,
        batch_iterations: int,
        policies_config: dict,
        policy_mapping_function,
        actor_keys: list,
        env,
        device: str = "cpu",
        id: int = -1,
        experiment_name = None,
    ):
        # TODO: config validation
        self.env = env
        self.id = id
        self.actor_keys = actor_keys
        self.batch_iterations = batch_iterations
        self.rollout_fragment_length = rollout_fragment_length
        self.batch_size = self.batch_iterations*self.rollout_fragment_length
        self.policy_keys = policies_config.keys()
        self.policy_mapping_function = policy_mapping_function
        self.experiment_name=experiment_name

        self.policies = {}
        self.memory = {}
        for key in self.policy_keys:
            self.policies[key] = build_policy(policies_config[key])
            self.memory[key] = RolloutBuffer()
        
        # TODO: add csv header

    # @exec_time
    def batch(self):
        """
        Creates a batch of `rollout_fragment_length` steps, save in `self.rollout_buffer`.
        """
        # FIXME: batch done in this way is wrong:
        # the env should be reset only when done==true, so we need
        # a common "self.obs" that is used at the beginning of the batch and
        # is set at the end (so we have save this obs for the next batch)
        # DONE: checked if this env returns `done` and YES, it DOES IT (at 1k steps - as config)

        # reset batching environment and get its observation
        obs = self.env.reset()

        # reset rollout_buffer
        for memory in self.memory.values():
            memory.clear()

        for counter in range(self.batch_size):
            # get actions, action_logprob for all agents in each policy* wrt observation
            policy_action, policy_logprob = self.get_actions(obs)

            # get new_observation, reward, done from stepping the environment
            next_obs, rew, done, _ = self.env.step(policy_action)

            if counter % self.rollout_fragment_length-1 == 0:
                # end this episode, start a new one. Set done to True (used in policies)
                # reset batching environment and get its observation
                done["__all__"] = True
                next_obs = self.env.reset()

            # save new_observation, reward, done, action, action_logprob in rollout_buffer
            for id in self.actor_keys:
                self.memory[self.policy_mapping_function(id)].update(
                    state=obs[id],
                    action=policy_action[id],
                    logprob=policy_logprob[id],
                    reward=rew[id],
                    is_terminal=done["__all__"],
                )

            obs = next_obs

        # Dump memory in file
        self.pickle_memory()

    # @exec_time
    def pickle_memory(self):
        """
        Dumps data in a file so it can be read by process' father
        """
        data_file = open(f'/dev/shm/{self.experiment_name}_{self.id}.bin', 'wb')
        pickle.dump(self.memory, data_file)
        data_file.close()

    def save_csv(self):
        """
        Append agent's total reward for this batch
        """

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
            (policy_action[key], policy_logprob[key]) = self.policies[
                self.policy_mapping_function(key)
            ].act(obs[key])

        return policy_action, policy_logprob

    def learn(self, memory):
        """
        TODO: docs
        """
        for key in self.policies:
            self.policies[key].learn(rollout_buffer=memory[key])

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
