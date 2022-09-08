# https://github.com/ray-project/ray/issues/3042
# https://github.com/ray-project/ray/pull/3051
# https://github.com/ray-project/ray/blob/master/rllib/models/preprocessors.py

import numpy as np

# TODO not efficient at all; find another way U_U
def dictToListNoChecks(data: dict) -> dict:
    dictionary = {}

    for values, keys in zip(data.values(), data.keys()):
        if type(values) == dict:    
            lista = np.empty(0)

            # Translates agent obs dict to np.array
            for d in values.values():
                lista = np.append(lista, d)
            
            dictionary[keys] = lista
        else:
            raise NotImplementedError(f"Unsupported type '{type(d)}'.")
        
    return dictionary


if __name__ == "__main__":
    _dizionario = {
        'a': {'a': np.zeros(2), 'b': np.ones(5)},
        'b': {'a': np.ones(4), 'b': np.zeros(1)}
    }

    _tradotto = dictToListNoChecks(_dizionario)
    print(_tradotto)
