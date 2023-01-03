"""
Multi-agent, multi-policy management algorithm.

It manages:
- batching
- giving actions to the environment
- each policy singularly so that it has all the required (correct) inputs
"""
from policies.policy_abs import Policy
from typing import Dict, Tuple
from utils.rollout_buffer import Memory


class Algorithm(object):
    """
    Multi-agent, multi-policy management algorithm.
    """

    def __init__(
        self,
        batch_size: int,
        policies: Dict[str, Policy],
        env,
    ):
        self.policies = policies
        self.batch_size = batch_size

        obs: Dict[str, _] = env.reset()

        self.policies_size = {}
        for key in policies.keys():
            self.policies_size[key] = 0
        available_agent_id = []
        for key in obs.keys():
            self.policies_size[self.policy_mapping_function(key)] += 1
            available_agent_id.append(key)

        self.memory = Memory(
            policy_mapping_fun=self.policy_mapping_function,
            available_agent_id=available_agent_id,
            batch_size=self.batch_size,
            policy_size=self.policies_size,
        )

    def policy_mapping_function(self, key: str) -> str:
        """
        It differenciates between two types of agents:
        `a` and `p`:    `a` -> economic player
                        `p` -> social planner

        Args:
            key: agent dictionary key
        """
        if str(key).isdigit() or key == "a":
            return "a"
        return "p"

    def train_one_step(self, env):
        """
        Train all policies.
        It creates a batch of size = `self.batch_size`, then
        this RolloutBuffer is splitted between each policy following
        `self.policy_mapping_fun` and trained respectivly to the
        corrisponding policy.

        Args:
            env: updated environment
        """
        self.memory.clear()

        self.batch(env)

        # FIXME: add management to get correct batch from the rollout_buffer
        # TODO: add logging -> can be done with returns
        for key in self.policies:
            self.policies[key].learn(
                rollout_buffer=self.memory.get[key],
                epochs=self.policies_size[key],
                steps_per_epoch=self.batch_size,
            )

    def batch(self, env):
        """
        Creates a batch of `self.batch_size` steps, save in `self.rollout_buffer`.

        Args:
            env: current environment
        """

        obs = env.reset()

        # TODO: here is possible getting average reward and stuff like that

        for i in range(self.batch_size):
            # Policies pick actions
            policy_action, policy_logprob = self.get_actions(obs)

            # Retrieve new state and reward
            next_obs, rew, done, _ = env.step(policy_action)

            self.memory.update(
                state=obs,
                action=policy_action,
                logprob=policy_logprob,
                reward=rew,
                is_terminal=done,
            )

            obs = next_obs

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
