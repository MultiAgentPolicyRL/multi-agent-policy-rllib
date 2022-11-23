"""
Set up and data check of algo's config
"""

import copy


class AlgorithmConfig():
    """
    a
    """
    def __init__(self, minibatch_size : int, policies_configs : dict, env, policy_mapping_fun=None, ):
        """
        Builds the config and validates data
        
        Arguments:
            minibatch_size: size of the minibatch # ADD logic
            n_actors: TODO automatic, dictionary with `mapping_fun_output`: n_elements_in_that_map
        
        """

        """
        - CHI vogliamo sia trainato e come 
        policies = {
            'a': policy_config(stuff)
            'p': policy_config(stuff)
        }
        """

        if policy_mapping_fun is None:
            self.policy_mapping_fun = self.policy_mapping_function
        else:
            self.policy_mapping_fun = policy_mapping_fun

        self.minibatch_size = minibatch_size
        self.policies_configs = policies_configs

        env = copy.deepcopy(env)
        obs : dict = env.reset()
        
        # dict containing `key` from `policy_mapping_fun` 
        # descrivere quanti agenti abbiamo per policy
        # nomi di tutti gli agenti
        agents_per_possible_policy = {}
        agents_name = []
        for key in obs.keys():
            agents_per_possible_policy[self.policy_mapping_fun(key)] = 0

        for key in obs.keys():
            agents_per_possible_policy[self.policy_mapping_fun(key)] += 1
            agents_name.append(self.policy_mapping_fun(key))
        self.agents_name = list(set(agents_name))

        # Calculate batch_size for each trained or untrained policy (follows policy_mapping_fun output)
        batch_size_per_policy = {}
        for key in agents_per_possible_policy:
            batch_size_per_policy[key] = minibatch_size//agents_per_possible_policy[key]

        batch_size = []
        # Calculate max batch size for trained policies:
        for key in batch_size_per_policy:
            if key in self.policies_configs:
                batch_size.append(batch_size_per_policy[key])

        # Maximum steps the environment actually has to do for batching
        # Each policy will get `batch_size` steps and will train it's model with `batch_size_per_policy[key]` elements
        self.batch_size = max(batch_size)

        # Set each policy `batch_size` fixed by n_agents in policy
        for key in self.policies_configs.keys():
            self.policies_configs[key].set_batch_size(batch_size_per_policy[key])



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