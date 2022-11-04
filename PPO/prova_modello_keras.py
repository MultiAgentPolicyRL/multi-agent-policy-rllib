from keras_model import build_model
from env_wrapper import RLlibEnvWrapper
from tf_models import KerasConvLSTM, get_flat_obs_size
import tensorflow as tf
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

env_config = {'env_config_dict': {
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
        ('Build', {
            'skill_dist':                   'pareto',
            'payment_max_skill_multiplier': 3,
            'build_labor':                  10,
            'payment':                      10
        }),
        # (2) Trading collectible resources
        ('ContinuousDoubleAuction', {
            'max_bid_ask':    10,
            'order_labor':    0.25,
            'max_num_orders': 5,
            'order_duration': 50
        }),
        # (3) Movement and resource collection
        ('Gather', {
            'move_labor':    1,
            'collect_labor': 1,
            'skill_dist':    'pareto'
        }),
        # (4) Planner
        ('PeriodicBracketTax', {
            'period':          100,
            'bracket_spacing': 'us-federal',
            'usd_scaling':     1000,
            'disable_taxes':   False
        })
    ],

    # ===== SCENARIO CLASS ARGUMENTS =====
    # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
    'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
    'starting_agent_coin': 10,
    'fixed_four_skill_and_loc': True,

    # ===== STANDARD ARGUMENTS ======
    # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
    'n_agents': 4,          # Number of non-planner agents (must be > 1)
    'world_size': [25, 25],  # [Height, Width] of the env world
    'episode_length': 1000,  # Number of timesteps per episode

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

    # How often to save the dense logs
    'dense_log_frequency': 1
}}

env = RLlibEnvWrapper(env_config)
obs = env.reset()


model = KerasConvLSTM(env.observation_space,
                      env.action_space, 50, model_config)
state = model.get_initial_state()

rank_1_tensor = tf.constant([50])
flat_obs_space = get_flat_obs_size(env.observation_space)


def dict_to_tensor_dict(a_dict: dict):
    tensor_dict = {}
    for key, value in a_dict.items():
        tensor_dict[key] = tf.convert_to_tensor(value, name=key)

    return tensor_dict

obs_tensor_dict = dict_to_tensor_dict(obs['0'])

input_dict = {
    'obs': obs_tensor_dict,
    # 'obs_flat': ,
    'prev_action': None, 
    'prev_reward': None, 
    'is_training': True
}

output, new_state = model.forward(input_dict, state, rank_1_tensor)
