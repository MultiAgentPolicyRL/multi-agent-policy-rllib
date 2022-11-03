# Inside a `global_trainer`:
- `fitness` -> it's a modified version of decision tree's fitness function
    - tf_model = kerasLSTMmodel
    - build_env (`env = env_wrapper(env_config)`)
    - build_ppo_trainer(`tf_model`) -> so it trains only agents ("id" is int)
    - `dt.new_episode`
    - obs = env.reset()
    - obs_pl_dt = filter(equalita', produttivita', risosrse [denaro, pietra, legno, forse media e std])(obs['p'])

    - **training loop**
        ### Get Actions
        - actions_pl_dt = dt(obs)
        - action_ppo = ppo.get_actions(obs)

        ### Step Env
        - actions = action_ppo + actions_pl_dt
        - obs, rew, done, info = env(actions)
        
        ### Train DT and PPO
        - dt.set_reward(rew["p"])
        - ppo.log_returns(rew)
        - ppo.train(env -> actual state, dt -> actual dt, used in minibatching)

    - return PPO weights, DT best tree

1. Implement PPO
    1. use custom keras model
    2. get `policies_to_train` and `policy_training_fn` 
    3. if obs is dict, split it and train as described by `policy_training_fn`
    4. it needs a `train_step` -> single step training (we are gonna call it instead of `train`).
    It's arguments will be, at least, `env` -> actual state, `dt` -> actual dt, used in minibatching.
    5. `act` function that returns `action_dict` -> as always if `obs` is dict split, else nope. 