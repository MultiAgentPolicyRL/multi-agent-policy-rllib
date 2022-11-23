"""
PPO's top level algorithm.
Manages batching and multi-agent training.
"""
import logging
import copy
import sys
from policy.policy import PPOAgent
from memory import BatchMemory
from algorithm.algorithm_config import AlgorithmConfig

class PpoAlgorithm(object):
    """
    PPO's top level algorithm.
    Manages batching and multi-agent training.
    """

    def __init__(
        self,
        algorithm_config : AlgorithmConfig
    ):
        """
        policy_config: dict,
        available_agent_groups,
        policy_mapping_fun=None

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
        self.algorithm_config = algorithm_config


        ###
        # Build dictionary for each policy (key) in `policy_config`
        ###
        self.training_policies = {}
        for key in self.algorithm_config.policies_configs:
            # config injection
            
            self.training_policies[key]: PPOAgent = PPOAgent(self.algorithm_config.policies_configs[key])
            

        # Setup batch memory
        # FIXME: doesn't work with `self.algorithm_config.policy_mapping_fun` reference
        self.memory = BatchMemory(
            self.algorithm_config.policy_mapping_function, self.algorithm_config.policies_configs, self.algorithm_config.agents_name
        )

    def train_one_step(self, env, obs = None):
        """
        Train all Policys
        Here PPO's Minibatch is generated and splitted to each policy, following
        `policy_mapping_fun` rules
        """
        # Resetting memory
        self.memory.reset_memory()

        steps = 0
        env = copy.deepcopy(env)
        state = env.reset()
        # state = obs

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
            # if steps % 1 == 0:
            #     logging.debug(f"step: {steps}")
            print(f"Steps: {steps+1}")
            steps += 1

        print(len(self.memory.batch['a']["states"]))
        print(len(self.memory.batch['a']["actions"]))
        print(len(self.memory.batch['a']["rewards"]))
        print(len(self.memory.batch['a']["predictions"]))
        print(len(self.memory.batch['a']["next_states"]))

        # sys.exit()

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
                ) = self.training_policies[self.algorithm_config.policy_mapping_function(key)].act(
                    obs[key]
                )
            else:
                # tmp to also feed the planner
                actions[key], actions_onehot[key], predictions[key] = (
                    [0, 0, 0, 0, 0, 0, 0],
                    0,
                    0,
                )
        # logging.debug(actions)
        return actions, actions_onehot, predictions
