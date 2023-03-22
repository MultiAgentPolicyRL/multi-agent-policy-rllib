"""
Rollout worker.Manages a policy and creates a batch.
"""
import logging
from typing import Tuple

from src.common import EmptyModel
from src.common.rollout_buffer import RolloutBuffer
from src.train.ppo import PpoPolicy, data_logging, save_batch
from src.train.ppo.utils.execution_time import exec_time

# pylint: disable=consider-using-dict-items,consider-iterating-dictionary


class RolloutWorker:
    """
    A lui arriva il solito policies_config che gia' contiene come sono
    impostate le policy e la loro configurazione.
    """

    def __init__(
        self,
        rollout_fragment_length: int,
        batch_iterations: int,
        policies_config: dict,
        mapping_function,
        actor_keys: list,
        env,
        seed: int,
        _id: int = -1,
        experiment_name=None,
    ):
        self.env = env
        self._id = _id

        self.actor_keys = actor_keys
        self.batch_iterations = batch_iterations
        self.rollout_fragment_length = rollout_fragment_length
        self.batch_size = self.batch_iterations * self.rollout_fragment_length
        self.policy_mapping_function = mapping_function
        self.experiment_name = experiment_name

        logging.debug(
            "ID: %s, Length: %s, iter:%s, batch_size:%s",
            self._id,
            self.rollout_fragment_length,
            self.batch_iterations,
            self.batch_size,
        )

        policy_keys = policies_config.keys()
        env.seed(seed + _id)
        env_keys = env.reset().keys()

        if self._id != -1:
            string = f"{','.join(map(str, env_keys))}\n"
            data_logging(data=string, experiment_id=self.experiment_name, id=self._id)
        else:
            string = (
                "a_actor_loss,a_critic_loss,a_entropy,p_a_loss,p_c_loss,p_entropy\n"
            )
            data_logging(data=string, experiment_id=self.experiment_name, id=self._id)

        # Build policices
        self.policies = {}
        for key in policy_keys:
            self.policies[key] = self._build_policy(policies_config[key])

        obs = env.reset()
        # self.memory = {}
        # self.rolling_memory = {}
        # for key0, key1 in zip(["0", "p"], ["a", "p"]):
        # self.memory[key1] = RolloutBuffer(obs[key0])
        # self.rolling_memory[key1] = RolloutBuffer(obs[key0])

        self.memory = RolloutBuffer(obs, self.policy_mapping_function)
        self.rolling_memory = RolloutBuffer(obs, self.policy_mapping_function)

        logging.debug("Rollout Worker %s built", self._id)

    def _build_policy(self, policy_config: dict):
        if policy_config["policy"] == EmptyModel:
            return EmptyModel(
                observation_space=policy_config["observation_space"],
                action_space=policy_config["action_space"],
            )
        elif policy_config["policy"] == PpoPolicy:
            return PpoPolicy(
                observation_space=policy_config["observation_space"],
                action_space=policy_config["action_space"],
                K_epochs=policy_config["k_epochs"],
                eps_clip=policy_config["eps_clip"],
                gamma=policy_config["gamma"],
                learning_rate=policy_config["learning_rate"],
                c1=policy_config["c1"],
                c2=policy_config["c2"],
                device=policy_config["device"],
                name=policy_config["name"],
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

        for i in range(self.batch_iterations):
            logging.debug(" ID: %s -- iteration: %s", self._id, i)

            for _ in range(self.rollout_fragment_length):
                # get actions, action_logprob for all agents in each policy* wrt observation
                policy_action, policy_logprob = self.get_actions(obs)

                # get new_observation, reward, done from stepping the environment
                next_obs, rew, done, _ = self.env.step(policy_action)

                if done["__all__"] is True:
                    next_obs = self.env.reset()

                """# save new_observation, reward, done, action, action_logprob in rollout_buffer
                # for _id in self.actor_keys:
                #     self.rolling_memory[self.policy_mapping_function(_id)].update(
                #         state=obs[_id],
                #         action=policy_action[_id],
                #         logprob=policy_logprob[_id],
                #         reward=rew[_id],
                #         is_terminal=done["__all__"],
                #     )"""

                self.rolling_memory.update(
                    action=policy_action,
                    logprob=policy_logprob,
                    state=obs,
                    reward=rew,
                    is_terminal=done["__all__"],
                )

                obs = next_obs

            """# logging.debug(
            #     "rolling memory-ID: %s - %s - iteration: %s",
            #     self._id,
            #     self.rolling_memory["a"].states["world-map"].shape,
            #     i,
            # )

            # for key in self.memory.keys():
            #     if key == "a" and self.memory["a"].states["world-map"] is not None:
            #         logging.debug(
            #             "before extend: ID %s - %s - iteration: %s",
            #             self._id,
            #             self.memory["a"].states["world-map"].shape,
            #             i,
            #         )

            #     self.memory[key].extend(self.rolling_memory[key], self._id)"""

            self.memory.extend(self.rolling_memory)
            self.rolling_memory.clear()
            """# logging.debug(
                # "After extend ID: %s - %s - iteration: %s",
                # self._id,
                # self.rolling_memory["a"].states["world-map"].shape,
                # i,
            # )
"""
        # Dump memory in ram
        save_batch(data=self.memory, worker_id=self._id)

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
        Call the learning function for each policy.
        """
        losses = []
        for key in self.policies:
            logging.debug("learning actor: %s", key)
            losses.append(self.policies[key].learn(rollout_buffer=memory[key]))

        rewards = []
        for _m in losses:
            for _k in _m:
                rewards.append(_k)

        data = f"{','.join(map(str, rewards))}\n"

        data_logging(data=data, experiment_id=self.experiment_name, id=self._id)

    def log_rewards(self):
        """
        Append agent's total reward for this batch
        """
        # FIXME: fix for new RolloutBuffer
        # data = [(self.memory["a"].rewards[i::4]) for i in range(4)]
        # data.append((self.memory["p"].rewards))

        # for i in range(len(data[4])):
            # splitted_data = [data[0][i], data[1][i], data[2][i], data[3][i], data[4][i]]
            # rewards = f"{','.join(map(str, splitted_data))}\n"
            # data_logging(data=rewards, experiment_id=self.experiment_name, id=self._id)

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

    def save_models(self):
        """
        Save the model of each policy.
        """
        for key in self.policies.keys():
            self.policies[key].save_model(
                "experiments/" + self.experiment_name + f"/models/{key}.pt"
            )

    def load_models(self, models_to_load: dict):
        """
        Load the model of each policy.

        It doesn't load 'p' policy.
        """
        for key in models_to_load.keys():
            self.policies[key].load_model(
                "experiments/" + models_to_load[key] + f"/models/{key}.pt"
            )

        logging.info("Models loaded!")
