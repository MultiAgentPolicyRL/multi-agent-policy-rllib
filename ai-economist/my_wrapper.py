"""
    DECORATOR

    Environment wrapper OpenAI-style.
    This wrapper adds the action and observation space to the environment,
    and adapts the reset, step, render functions to be like OpenAI-SingleAgentEnv
"""

from ai_economist import foundation
import numpy as np

"""
    Wrapper scheme:

    from numpy to dict
    steps/function
    from dict to numpy
"""

def env_decorator(env):
    """
    TODO
    probabilmente basta che passo al training per gli step semplicemente rewards e action_spaces degli agenti + planner.
    Guardare file 'economic_simulation_basic'
    """
    
    
    pass
