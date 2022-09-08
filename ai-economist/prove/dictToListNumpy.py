# https://github.com/ray-project/ray/issues/3042
# https://github.com/ray-project/ray/pull/3051
# https://github.com/ray-project/ray/blob/master/rllib/models/preprocessors.py

import numpy as np

def dictToList(data: dict):
    lista = np.empty(0)

    for d in data.values():

        if type(d) == np.ndarray:
            lista = np.append(lista, d)
        else:
            # NOT IMPLEMENTED
            print("PORCODDIO")
    return lista

def dictToListNoChecks(data: dict):
    lista = np.empty(0)
    for d in data.values():
        lista = np.append(lista, d)
    return lista

if __name__ == "__main__":
    listaUno = np.ones(3)
    listaZero = np.zeros(2)
    listaVal = np.array([10,20])

    dizionario = {
        'a': listaUno,
        'b': listaZero,
        'c': listaVal
    }

    lista = dictToList(dizionario)
    lista = dictToListNoChecks(dizionario)
    print(lista)

#     def dictToList(data: dict):
#     lista = []

#     for d in data.values():
#         if type(d) == list:
#             lista = lista + d    
#         elif type(d) != dict:
#             lista.append(d)
#         elif type(d) == dict:
#             lista += dictToList(d)
#     return lista        

# data = {'a': 1, 'b':2, 'c': [1,2,3,4,5], 'd': {'e': 123, 'f': [11,22,33,44,55], 'g': {'h': 12346789}}}

# data = dictToList(data)
# data =  np.asarray(data)

# print(f"{data}, datatype {type(data)} ")