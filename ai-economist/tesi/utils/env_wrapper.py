"""
Wrapper for making the gather-trade-build environment an OpenAI compatible environment.
"""

import numpy as np
from ai_economist import foundation
from gym.utils import seeding
import random


## OpenAI MultiAgentEnvironment
## https://github.com/openai/multiagent-particle-envs/blob/master/multiagent/environment.py

# constants
_BIG_NUMBER = 1e20

def __recursive_list_to_np_array(d):
    """
    If d is a dictionary, recursively convert all lists to numpy arrays

    Args:
        d: a dictionary
    
    Returns:
        d: a dictionary with all lists converted to numpy arrays
    """
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, list):
                new_d[k] = np.array(v)
            elif isinstance(v, dict):
                new_d[k] = __recursive_list_to_np_array(v)
            elif isinstance(v, (float, int, np.floating, np.integer)):
                new_d[k] = np.array([v])
            elif isinstance(v, np.ndarray):
                new_d[k] = v
            else:
                raise AssertionError
        return new_d
    raise AssertionError

def __dict_to_spaces_dict(self, obs):
        dict_of_spaces = {}
        for k, v in obs.items():

            # list of lists are listified np arrays
            _v = v
            if isinstance(v, list):
                _v = np.array(v)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                _v = np.array([v])

            # assign Space
            if isinstance(_v, np.ndarray):
                x = float(_BIG_NUMBER)
                # Warnings for extreme values
                if np.max(_v) > x:
                    warnings.warn("Input is too large!")
                if np.min(_v) < -x:
                    warnings.warn("Input is too small!")
                box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                # This loop avoids issues with overflow to make sure low/high are good.
                while not low_high_valid:
                    x = x // 2
                    box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                    low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                dict_of_spaces[k] = box

            elif isinstance(_v, dict):
                dict_of_spaces[k] = self._dict_to_spaces_dict(_v)
            else:
                raise TypeError
        return spaces.Dict(dict_of_spaces)


def pretty_print(dictionary):
    for key in dictionary:
        print(f"{key:15s}: {dictionary[key].shape}")
    print("\n")

def EnvWrapper():
    """
    Environment wrapper OpenAI-style. -> make as a decorator
    This wrapper adds the action and observation space to the environment,
    and adapts the reset, step, render functions to be like OpenAI-SingleAgentEnv
    """
    
    """
    IN PRATICA VOGLIO WRAPPARE L'ENV FACENDO IN MODO CHE I DATI CHE MI DANNO QUELLI DI salesforce siano ADEGUATI a quello
    che richiede il modulo di training
    prendo spunto da qua: https://github.com/rhklite/Parallel-PPO-PyTorch
    MANDO A FANCULO LA VETTORIZZAZIONE (chissenefrega, tanto e' "inutile")
    

    
    
    L'idea di fare un wrapper la abbandono adesso.

    LOL ->> stronzata ->> probabilmente fare modificare il blocco di RLLib e' un casino assurdo.
    Visto che la loro idea di fare multithread e' di poter lanciare piu esperimenti in 
    contemporanea e che il wrapper di ai_economist gia' lo permette -> 
    vedere come hanno implementato PPO e buttarci dentro l'algoritmo del prof

    """
    

