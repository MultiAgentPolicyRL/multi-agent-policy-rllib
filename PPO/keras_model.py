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


    _, v = obs_space.spaces.items[0]
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