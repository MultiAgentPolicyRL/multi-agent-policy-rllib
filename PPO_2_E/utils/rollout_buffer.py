"""
    Buffer used to store batched data
"""
# FIXME: doesn't work with multi-agent ('a', 'p')
# FIXME: doesn't work with lstm (multi-agent data is shuffled)
class RolloutBuffer:
    """
    Buffer used to store batched data
    """

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        """
        Clear the buffer
        """
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
