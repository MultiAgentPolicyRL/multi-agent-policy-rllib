from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env_wrapped_tmp import EnvWrapper
import numpy as np

# Same environment as economic_simulation_basia.ipynb

# Define the configuration of the environment that will be built

env_config = {'env_config_dict' : {
    
    'scenario_name': 'layout_from_file/simple_wood_and_stone',
    
    'components': [
        # (1) Building houses
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        # (2) Trading collectible resources
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        # (3) Movement and resource collection
        ('Gather', {}),
    ],
    
    'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
    'starting_agent_coin': 10,
    'fixed_four_skill_and_loc': True,
    
    'n_agents': 2,          # Number of non-planner agents (must be > 1)
    'world_size': [25, 25], # [Height, Width] of the env world
    'episode_length': 1000, # Number of timesteps per episode
    
    'multi_action_mode_agents': False,
    'multi_action_mode_planner': True,
    
    'flatten_observations': True,

    'flatten_masks': True,
    }
}

env = EnvWrapper(env_config)
env.reset()
# obs,rew,done,_ = env.step({
# '0':1,
# '1':1
# })

# print(f"Type of obs: {type(obs)}, Type of rew: {type(rew)}, Type of done: {type(done)}")

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=2)
