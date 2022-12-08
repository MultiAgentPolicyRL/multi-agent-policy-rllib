"""
Set up and data check of algo's config
"""
import copy


class AlgorithmConfig:
    """
    a
    """

    def __init__(
        self,
        minibatch_size: int,
        policies_configs: dict,
        env,
        seed: int,
        policy_mapping_fun=None,
        multiprocessing: bool = False,
        num_workers: int = 0,
    ):
        """
        Builds the config and validates data

        Arguments:
            minibatch_size: size of the minibatch # ADD logic
            n_actors: TODO automatic, dictionary with `mapping_fun_output`: n_elements_in_that_map
            multiprocessing: if you want multiprocessing
            num_workers: works only with `multiprocessing` active
        """

        """
        - CHI vogliamo sia trainato e come 
        policies = {
            'a': policy_config(stuff)
            'p': policy_config(stuff)
        }
        """
        self.num_workers = num_workers
        self.multiprocessing = multiprocessing
        self.seed = seed
        if policy_mapping_fun is None:
            self.policy_mapping_fun = self.policy_mapping_function
        else:
            self.policy_mapping_fun = policy_mapping_fun

        self.minibatch_size = minibatch_size
        self.policies_configs = policies_configs

        self.env = copy.deepcopy(env)
        obs: dict = env.reset()

        # dict containing `key` from `policy_mapping_fun`
        # descrivere quanti agenti abbiamo per policy
        # nomi di tutti gli agenti
        self.agents_per_possible_policy = {}
        self.agents_name = []
        for key in obs.keys():
            self.agents_per_possible_policy[self.policy_mapping_fun(key)] = 0

        for key in obs.keys():
            self.agents_per_possible_policy[self.policy_mapping_fun(key)] += 1
            self.agents_name.append(key)

        self.batch_size = self.minibatch_size

        for key in self.policies_configs.keys():
            self.policies_configs[key].set_batch_size_and_agents_per_possible_policy(
                self.batch_size, self.agents_per_possible_policy[key], self.num_workers
            )

        # Checks if multiprocessing and num_workers makes no sense
        if (self.multiprocessing and self.num_workers != 0) or (
            self.multiprocessing is False and self.num_workers != 0
        ):
            if self.num_workers < 0:
                ValueError("`num_workers` must be > 0!")
            AttributeError(
                "Multiprocessing and num_workers work together, put correct values. Ex: True, 4"
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
