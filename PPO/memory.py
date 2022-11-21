class BatchMemory:
    def __init__(
        self, policy_mapping_function, policy_config: dict, available_agent_groups: list
    ):
        self.policy_mapping_function = policy_mapping_function

        # states, next_states, actions, rewards, predictions, dones
        self.batch = {}
        keys = ["states", "next_states", "actions", "rewards", "predictions"]

        # Initialize empty memory
        for key in available_agent_groups:
            self.batch[key] = {e: [] for e in keys}

        """
        Structure is gonna be like:
        batch: {
            'a': [obs1, obs2, obs3]
            'p': [obsp1, obsp2, obsp3]
        }

        where obs1,2,3 are like
        { '0': dict
          '1': dict
          'n': dict 
        }

        end obsp1,2,3 : {
            'p' : dict
        }
        """

    def reset_memory(self):
        """
        Method.
        Clears the memory.
        """
        for key in self.batch.keys():
            policy = self.policy_mapping_function(key)
            # print(f"Appending to {policy} element of: {key}")

            self.batch[policy]["states"].clear()
            self.batch[policy]["next_states"].clear()
            self.batch[policy]["actions"].clear()
            self.batch[policy]["rewards"].clear()
            self.batch[policy]["predictions"].clear()

    def update_memory(
        self,
        state: dict,
        next_state: dict,
        action_onehot: dict,
        reward: dict,
        prediction: dict,
    ):
        """
        Method.
        Append NEW states, next_states, actions, rewards, predictions to the memory
        splitting them between all policies.
        Plit done by `policy_mapping_function` classification.

        Arguments:
            state:
            next_state:
            action_onehot:
            reward:
            prediction:

        Returns:
            nothing
        """
        for key in state.keys():
            policy = self.policy_mapping_function(key)
            # print(f"Appending to {policy} element of: {key}")

            self.batch[policy]["states"].append(state[key])
            self.batch[policy]["actions"].append(action_onehot[key])
            self.batch[policy]["rewards"].append(reward[key])
            self.batch[policy]["predictions"].append(prediction[key])
            self.batch[policy]["next_states"].append(next_state[key])

    # DEP
    def get_batch_policy(self, key: str):
        """
        returns `states, actions, rewards, predictions, next_states` for the specified policy
        """
        return (
            self.batch[key]["states"],
            self.batch[key]["actions"],
            self.batch[key]["rewards"],
            self.batch[key]["predictions"],
            self.batch[key]["next_states"],
        )
