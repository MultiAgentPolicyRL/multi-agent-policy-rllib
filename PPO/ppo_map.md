# Multi-Agent PPO thoughts
### In Italian 
TODO: in english

- Algoritmo top level - PPO (qui **non** verra' introdotto il DT)
  - Avra':
    - `policy_mapping_function`   
        restituisce `a` o `p` in base alla chiave che gli viene passata
    - `policy_to_train`   
        dizionario con `a`,`p` e le rispettive policy
        ```
            `policy_to_train` : {
                `a`: ppo_core(),
                `p`: ppo_core(),
            }
        ```
  - Gestisce:
    - il training per policy singola:
      - stepping cloned_env (mini-batch)
      - separazione obs,rew per policy, le mette in liste  
  
        ``` 
            obs_dict: {
                'a': [...],
                'p': [...],
            }
        ```
        uguale per le rew
      - chiamata del `train_one_step_batching` (quindi del training)
        - forse si puo' rimuovere `train_one_step_batching` e lasciare solamente `_learn` (farlo diventare un metodo esterno) visto che i dati di batch vengono gia' forniti
      - costruisce il dizionario delle actions: `act`
        - chiama singolarmente le policy (`a`, `p`)
        - costruisce il dict dai risultati
            ```
                actions: {
                    '0': [...],
                    '1': [...],
                    '2': [...],
                    ...
                    'p': [...]
                }
            ```
        - questo dizionario viene creato "senza far sapere alla policy singola" che e' multiagente (pseudo codice brutto):
            ```
                obs # environment obs, dict of obs_per_agent
                actions = {}
                for agent in obs:
                    which_policy = mapping(agent)
                        actions[agent]=policy[which_policy].act(obs[agent])
                actions is ready to be used for stepping
            ```
        - per pulizia del codice la funzione che passa i dizionari a tensori viene tenuta nella "ppo_singola"

  - Espone:
    - train(env)
      - fa una copia dell'env che gli viene passato e ci fa batching
    - act
      - restituisce il dizionario delle azioni