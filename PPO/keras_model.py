import tensorflow as tf
import keras.layers as k
from keras.models import Model


def feed_model(obs, model):
    """ Takes in input an observation (for a single agent (e.g., obs['0'])) and a model and returns the output. """
    numerical_features = ['world-inventory-Coin',
                          'world-inventory-Wood',
                          'world-inventory-Stone',
                          'Build-build_payment',
                          'Build-build_skill',
                          'ContinuousDoubleAuction-market_rate-Stone',
                          'ContinuousDoubleAuction-market_rate-Wood',
                          'ContinuousDoubleAuction-market_rate-Stone',
                          'ContinuousDoubleAuction-price_history-Stone',
                          'ContinuousDoubleAuction-available_asks-Stone',
                          'ContinuousDoubleAuction-available_bids-Stone',
                          'ContinuousDoubleAuction-my_asks-Stone',
                          'ContinuousDoubleAuction-my_bids-Stone',
                          'ContinuousDoubleAuction-market_rate-Wood',
                          'ContinuousDoubleAuction-price_history-Wood',
                          'ContinuousDoubleAuction-available_asks-Wood',
                          'ContinuousDoubleAuction-available_bids-Wood',
                          'ContinuousDoubleAuction-my_asks-Wood',
                          'ContinuousDoubleAuction-my_bids-Wood']

    # import pdb
    # pdb.set_trace()
    return obs['action_mask'] * model(obs['world-map'], numerical_features)


def _get_base_model(obs_space):
    """ Builds the model. Takes in input the parameters that were not specified in the paper. """
    # Get Conv Shape
    conv_shape_r = None
    conv_shape_c = None
    conv_map_channels = None
    conv_idx_channels = None
    generic_name = None

    # FIXME here
    _, v = obs_space.spaces.items
    conv_shape_r, conv_shape_c, conv_map_channels = (
        v.shape[1], v.shape[2], v.shape[0], )
    conv_idx_channels = v.shape[0] * 4  # idx_emb_dim

    conv_shape = (
        conv_shape_r,
        conv_shape_c,
        conv_map_channels + conv_idx_channels,
    )

    # Build model
    map_cnn = k.Input(shape=(15, 15, 7))
    map_cnn = k.Conv2D(
        16, (3, 3), strides=2, activation='relu', input_shape=conv_shape)(map_cnn)
    map_cnn = k.Conv2D(
        32, (3, 3), strides=2, activation='relu')(map_cnn)

    map_cnn = k.Flatten()(map_cnn)

    info_input = k.Input(shape=(56*2+5))
    mlp1 = k.Concatenate([map_cnn, info_input])
    mlp1 = k.Dense(128, activation='relu')(mlp1)
    mlp1 = k.Dense(128, activation='relu')(mlp1)

    lstm = k.LSTM(128)(mlp1)
    mlp2 = k.Dense(50)(lstm)
    return mlp2


def build_model(obs_shape, action_space):
    # state = k.Input(shape=obs_shape)

    vf = _get_base_model(obs_space=obs_shape)
    value_pred = k.Dense(1, name="Out_value")(vf)

    pi = _get_base_model(obs_space=obs_shape)
    action_probs = k.Dense(
        action_space, name="Out_probs", activation='relu')(pi)

    model = Model(inputs=obs_shape, output=[action_probs, value_pred])



if __name__=='__main__':
    from env_wrapper import RLlibEnvWrapper

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

    
    model = build_model(env.observation_space, env.action_space)