import tensorflow as tf
import numpy as np
from keras_model_base import feed_model, get_model
from ai_economist import foundation
from keras import Model

from tensorflow.python.framework.ops import enable_eager_execution
enable_eager_execution()

model_config = {
    'custom_model': "keras_conv_lstm",
    'custom_options': {
        'fc_dim': 128,
        'idx_emb_dim': 4,
        'input_emb_vocab': 100,
        'lstm_cell_size': 128,
        'num_conv': 2,
        'num_fc': 2,
    },
    'max_seq_len': 25,
}

env_config_wrapper = {
    'env_config_dict': {
        # ===== SCENARIO CLASS =====
        # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
        # The environment object will be an instance of the Scenario class.
        'scenario_name':
        'layout_from_file/simple_wood_and_stone',

        # ===== COMPONENTS =====
        # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
        #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
        #   {component_kwargs} is a dictionary of kwargs passed to the Component class
        # The order in which components reset, step, and generate obs follows their listed order below.
        'components': [
            # (1) Building houses
            ('Build', {
                'skill_dist': 'pareto',
                'payment_max_skill_multiplier': 3,
                'build_labor': 10,
                'payment': 10
            }),
            # (2) Trading collectible resources
            ('ContinuousDoubleAuction', {
                'max_bid_ask': 10,
                'order_labor': 0.25,
                'max_num_orders': 5,
                'order_duration': 50
            }),
            # (3) Movement and resource collection
            ('Gather', {
                'move_labor': 1,
                'collect_labor': 1,
                'skill_dist': 'pareto'
            }),
            # (4) Planner
            ('PeriodicBracketTax', {
                'period': 100,
                'bracket_spacing': 'us-federal',
                'usd_scaling': 1000,
                'disable_taxes': False
            })
        ],

        # ===== SCENARIO CLASS ARGUMENTS =====
        # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
        'env_layout_file':
        'quadrant_25x25_20each_30clump.txt',
        'starting_agent_coin':
        10,
        'fixed_four_skill_and_loc':
        True,

        # ===== STANDARD ARGUMENTS ======
        # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
        'n_agents':
        4,  # Number of non-planner agents (must be > 1)
        'world_size': [25, 25],  # [Height, Width] of the env world
        'episode_length':
        1000,  # Number of timesteps per episode

        # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
        # Otherwise, the policy selects only 1 action.
        'multi_action_mode_agents':
        False,
        'multi_action_mode_planner':
        True,

        # When flattening observations, concatenate scalar & vector observations before output.
        # Otherwise, return observations with minimal processing.
        'flatten_observations':
        True,
        # When Flattening masks, concatenate each action subspace mask into a single array.
        # Note: flatten_masks = True is required for masking action logits in the code below.
        'flatten_masks':
        True,

        # How often to save the dense logs
        'dense_log_frequency':
        1
    }
}

env_config = {
    # ===== SCENARIO CLASS =====
    # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
    # The environment object will be an instance of the Scenario class.
    'scenario_name': 'layout_from_file/simple_wood_and_stone',
    
    # ===== COMPONENTS =====
    # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
    #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
    #   {component_kwargs} is a dictionary of kwargs passed to the Component class
    # The order in which components reset, step, and generate obs follows their listed order below.
    'components': [
        # (1) Building houses
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        # (2) Trading collectible resources
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        # (3) Movement and resource collection
        ('Gather', {}),
    ],
    
    # ===== SCENARIO CLASS ARGUMENTS =====
    # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
    'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
    'starting_agent_coin': 10,
    'fixed_four_skill_and_loc': True,
    
    # ===== STANDARD ARGUMENTS ======
    # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
    'n_agents': 4,          # Number of non-planner agents (must be > 1)
    'world_size': [25, 25], # [Height, Width] of the env world
    'episode_length': 1000, # Number of timesteps per episode
    
    # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
    # Otherwise, the policy selects only 1 action.
    'multi_action_mode_agents': False,
    'multi_action_mode_planner': True,
    
    # When flattening observations, concatenate scalar & vector observations before output.
    # Otherwise, return observations with minimal processing.
    'flatten_observations': True,
    # When Flattening masks, concatenate each action subspace mask into a single array.
    # Note: flatten_masks = True is required for masking action logits in the code below.
    'flatten_masks': True,
}

# def to_shape(a, shape):
#     y_, x_ = shape
#     y, x = a.shape
#     y_pad = (y_ - y)
#     x_pad = (x_ - x)
#     return np.pad(a, ((y_pad // 2, y_pad // 2 + y_pad % 2),
#                       (x_pad // 2, x_pad // 2 + x_pad % 2)),
#                   mode='constant')


def dict_to_tensor_dict(a_dict: dict):
    """
    pass a single agent obs, returns it's tensor_dict
    """
    tensor_dict = {}
    for key, value in a_dict.items():
        tensor_dict[key] = tf.convert_to_tensor(value, name=key)
        tensor_dict[key] = tf.expand_dims(tensor_dict[key], axis=0)
        
    return tensor_dict 

if __name__ == '__main__':
    env = foundation.make_env_instance(**env_config_wrapper['env_config_dict']) #Edited
    #env = foundation.make_env_instance(**env_config) # Original
    obs = env.reset()

    model: Model = get_model([16, 32], 3)

    for i in range(100):
        actions_dict = {}
        for agent_id in obs.keys():
            if agent_id != 'p':
                actions_dict[agent_id] = feed_model(dict_to_tensor_dict(obs[agent_id]), model)
            else:
                actions_dict['p'] = np.array([0 for _ in range(env.get_agent('p')._unique_actions)])

        obs, rewards, dones, infos = env.step(actions_dict)
        print(f"Rewards: {rewards}, dones: {dones}, infos: {infos}")