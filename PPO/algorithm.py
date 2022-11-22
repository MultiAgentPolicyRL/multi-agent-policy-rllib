"""
PPO's top level algorithm.
Manages batching and multi-agent training.
"""
import logging
import copy
import sys
from policy import PPOAgent
from memory import BatchMemory


class PpoAlgorithm(object):
    """
    PPO's top level algorithm.
    Manages batching and multi-agent training.
    """

    def __init__(
        self,
        policy_config: dict,
        available_agent_groups,
        policy_mapping_fun=None,
    ):
        """
        policy config can be like:

        policy_config: {
            'a' : config or None,
            'p' : config or None,
        }
        """
        ##############
        # TMP STUFF HERE
        # all this needs to be auto
        self.minibatch_size = 50
        self.n_actors = {
            'a': 4,
            'p': 1
        }
        ###############

        if policy_mapping_fun:
            self.policy_mapping_fun = policy_mapping_fun
        else:
            self.policy_mapping_fun = (self.policy_mapping_function,)

        self.available_agent_groups = available_agent_groups

        self.training_policies = {}

        for key in policy_config:
            # config injection
            if policy_config[key] is not None:
                self.training_policies[key]: PPOAgent = PPOAgent(
                    policy_config=policy_config[key], batch_size=self.minibatch_size # //self.n_actors[key]
                )
            else:
                # USING THIS RN
                self.training_policies[key]: PPOAgent = PPOAgent(batch_size=self.minibatch_size)

        # Setup batch memory
        # FIXME: doesn't work with `self.policy_mapping_fun` reference
        self.memory = BatchMemory(
            self.policy_mapping_function, policy_config, available_agent_groups
        )

    def policy_mapping_function(self, key: str) -> str:
        """
        Use it by passing keys of a dictionary to differentiate between agents

        default for ai-economist environment:
        returns a if `key` is a number -> if the key of the dictionary is a number,
        returns p if `key` is a string -> social planner
        """
        if str(key).isdigit() or key == "a":
            return "a"
        return "p"

    def train_one_step(self, env, obs = None):
        """
        Train all Policys
        Here PPO's Minibatch is generated and splitted to each policy, following
        `policy_mapping_fun` rules
        """
        # Resetting memory
        self.memory.reset_memory()

        steps = 0
        # env = copy.deepcopy(env)
        # state = env.reset()
        state = obs

        # Collecting data for batching
        logging.debug("Batching")
        while steps < self.minibatch_size//4:
            # Actor picks an action
            action, action_onehot, prediction = self.get_actions(state)

            # Retrieve new state, rew
            next_state, reward, _, _ = env.step(action)

            # Memorize (state, action, reward) for trainig
            self.memory.update_memory(
                state, next_state, action_onehot, reward, prediction
            )
            if steps % 1 == 0:
                logging.debug(f"step: {steps}")

            steps += 1

        # Pass batch to the correct policy to perform training
        for key in self.training_policies:
            logging.debug(f"Training policy {key}")
            self.training_policies[key].learn(
                self.memory.batch[key]["states"],
                self.memory.batch[key]["actions"],
                self.memory.batch[key]["rewards"],
                self.memory.batch[key]["predictions"],
                self.memory.batch[key]["next_states"],
            )

        return env

    def get_actions(self, obs: dict) -> dict:
        """
        Build action dictionary from env observations. Output has thi structure:

                actions: {
                    '0': [...],
                    '1': [...],
                    '2': [...],
                    ...
                    'p': [...]
                }

        FIXME: Planner

        Arguments:
            obs: observation dictionary of the environment, it contains all observations for each agent

        Returns:
            actions dict: actions for each agent
        """

        actions, actions_onehot, predictions = {}, {}, {}
        for key in obs.keys():
            if key != "p":
                # print(self._policy_mapping_function(key))
                (
                    actions[key],
                    actions_onehot[key],
                    predictions[key],
                ) = self.training_policies[self.policy_mapping_function(key)].act(
                    obs[key]
                )
            else:
                # tmp to also feed the planner
                actions[key], actions_onehot[key], predictions[key] = (
                    [0, 0, 0, 0, 0, 0, 0],
                    0,
                    0,
                )

        return actions, actions_onehot, predictions
