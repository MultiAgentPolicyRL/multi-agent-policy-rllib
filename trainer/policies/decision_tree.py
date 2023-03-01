import numpy as np

from ai_economist_ppo_dt.torch.models.actor_critic import LSTMModel


def string_to_dict(x):
    """
    This function splits a string into a dict.
    The string must be in the format: key0-value0#key1-value1#...#keyn-valuen
    """
    result = {}
    items = x.split("#")

    for i in items:
        key, value = i.split("-")
        try:
            result[key] = int(value)
        except:
            try:
                result[key] = float(value)
            except:
                result[key] = value

    return result


class DecisionTree:
    def __init__(self):
        self.current_reward = 0
        self.last_leaf = None

    # @abc.abstractmethod
    # def get_action(self, input):
    #     pass

    def set_reward(self, reward):
        self.current_reward = reward

    def new_episode(self):
        self.last_leaf = None


class Leaf:
    def get_action(self):
        pass

    def update(self):
        pass


class QLearningLeaf(Leaf):
    def __init__(self, rl_model: LSTMModel):  # , learning_rate, discount_factor):
        self.rl_model = rl_model

        # self.learning_rate = learning_rate
        # self.discount_factor = discount_factor
        self.parent = None

        n_actions = rl_model.output_size
        self.iteration = [1] * n_actions

        self.q = np.zeros(n_actions, dtype=np.float32)
        self.last_action = 0
        self.last_logits = None
        self.last_value = 0

    def get_action(self, x):
        # action = np.argmax(self.q)

        # This is already done in parent
        # self.last_action = action
        return self.rl_model(x)

    def update(self, qprime):
        if self.last_action is not None:
            for action in self.last_action:
                self.q[int(action.item())] += qprime

    def next_iteration(self):
        for action in self.last_action:
            self.iteration[int(action.item())] += 1

    def __repr__(self):
        return ", ".join(["{:.2f}".format(k) for k in self.q])

    def __str__(self):
        return repr(self)


class EpsGreedyLeaf(QLearningLeaf):
    def __init__(self, rl_model: LSTMModel):
        super().__init__(rl_model=rl_model)

    def get_action(self, x):
        action, policy, value = super().get_action(x)

        self.last_action = action
        self.last_logits = policy
        self.last_value = value

        self.next_iteration()

        return action, policy, value


class RandomlyInitializedEpsGreedyLeaf(EpsGreedyLeaf):
    def __init__(self, rl_model: LSTMModel):
        """
        Initialize the leaf.
        Params:
            - n_actions: The number of actions
            - learning_rate: the learning rate to use, callable or float
            - discount_factor: the discount factor, float
            - epsilon: epsilon parameter for the random choice
            - low: lower bound for the initialization
            - up: upper bound for the initialization
        """
        super(RandomlyInitializedEpsGreedyLeaf, self).__init__(rl_model=rl_model)


class CLeaf(RandomlyInitializedEpsGreedyLeaf):
    def __init__(self, rl_model, *args, **kwargs):
        super(CLeaf, self).__init__(rl_model=rl_model)


class PythonDT(DecisionTree):
    def __init__(self, phenotype, rl_model: LSTMModel):
        super(PythonDT, self).__init__()
        self.program = phenotype
        self.leaves = {}
        n_leaves = 0

        while "_leaf" in self.program:
            new_leaf = CLeaf(rl_model=rl_model)
            leaf_name = "leaf_{}".format(n_leaves)
            self.leaves[leaf_name] = new_leaf

            self.program = self.program.replace(
                "_leaf", "'{}.get_action()'".format(leaf_name), 1
            )
            self.program = self.program.replace("_leaf", "{}".format(leaf_name), 1)

            n_leaves += 1
        self.exec_ = compile(self.program, "<string>", "exec", optimize=2)

    def get_action(self, x: dict):
        if len(self.program) == 0:
            return None
        variables = {}  # {"out": None, "leaf": None}

        _in = x.get("flat", None).squeeze()
        for idx, value in enumerate(_in):
            variables["_in_{}".format(idx)] = value
        variables.update(self.leaves)

        exec(self.exec_, variables)

        current_leaf: CLeaf = self.leaves[variables["leaf"]]

        # current_q_value = max(current_leaf.q) # Here goes the value predicted from the model
        # if self.last_leaf is not None:
        #     self.last_leaf.update(self.current_reward, current_q_value)

        self.last_leaf = current_leaf

        action, logit, value = current_leaf.get_action(x)

        current_q_value = value
        if self.last_leaf is not None:
            self.last_leaf.update(current_q_value)

        if action is None:
            print("[DECISION TREE] action is None")
        if logit is None:
            print("[DECISION TREE] logit is None")
        if value is None:
            print("[DECISION TREE] value is None")

        return action, logit, value

    def __call__(self, x):
        return self.get_action(x)

    def __str__(self):
        return self.program
