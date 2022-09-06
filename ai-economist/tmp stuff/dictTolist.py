# https://github.com/ray-project/ray/issues/3042
# https://github.com/ray-project/ray/pull/3051
# https://github.com/ray-project/ray/blob/master/rllib/models/preprocessors.py

import numpy as np
def dictToList(data: dict):
    lista = []

    for d in data.values():
        if type(d) == list:
            lista = lista + d    
        elif type(d) != dict:
            lista.append(d)
        elif type(d) == dict:
            lista += dictToList(d)
    return lista        

data = {'a': 1, 'b':2, 'c': [1,2,3,4,5], 'd': {'e': 123, 'f': [11,22,33,44,55], 'g': {'h': 12346789}}}

data = dictToList(data)
data =  np.asarray(data)

print(f"{data}, datatype {type(data)} ")
