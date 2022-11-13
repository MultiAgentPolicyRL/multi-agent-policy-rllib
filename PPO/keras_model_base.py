import tensorflow as tf
import keras.layers as k
from keras import Model


def feed_model(obs, model):
    """ Takes in input an observation (for a single agent (e.g., obs['0'])) and a model and returns the output. """
    numerical_features = []

    for data in ['flat']:
        numerical_features.append(obs[data])

    # for x in ['world-inventory-Coin',
    #           'world-inventory-Wood',
    #           'world-inventory-Stone',
    #           'Build-build_payment',
    #           'Build-build_skill',
    #           'ContinuousDoubleAuction-market_rate-Stone',
    #           'ContinuousDoubleAuction-market_rate-Wood']:
    #     numerical_features.append(obs[x])

    # for x in ['ContinuousDoubleAuction-market_rate-Stone',
    #           'ContinuousDoubleAuction-price_history-Stone',
    #           'ContinuousDoubleAuction-available_asks-Stone',
    #           'ContinuousDoubleAuction-available_bids-Stone',
    #           'ContinuousDoubleAuction-my_asks-Stone',
    #           'ContinuousDoubleAuction-my_bids-Stone',
    #           'ContinuousDoubleAuction-market_rate-Wood',
    #           'ContinuousDoubleAuction-price_history-Wood',
    #           'ContinuousDoubleAuction-available_asks-Wood',
    #           'ContinuousDoubleAuction-available_bids-Wood',
    #           'ContinuousDoubleAuction-my_asks-Wood',
    #           'ContinuousDoubleAuction-my_bids-Wood']:
    #     numerical_features.extend(obs[x])

    #print([obs['world-map']])
    return obs['action_mask'] * model([obs['world-map'], [numerical_features]])


def get_model(conv_filters, filter_size):
    """ Builds the model. Takes in input the parameters that were not specified in the paper. """
    cnn_in = k.Input(shape=(7, 11, 11))
    map_cnn = k.Conv2D(conv_filters[0], filter_size, activation='relu')(cnn_in)
    map_cnn = k.Conv2D(conv_filters[1], filter_size, activation='relu')(map_cnn)
    map_cnn = k.Flatten()(map_cnn)

    info_input = k.Input(shape=(136))
    mlp1 = k.Concatenate()([map_cnn, info_input])
    mlp1 = k.Dense(128, activation='relu')(mlp1)
    mlp1 = k.Dense(128, activation='relu')(mlp1)
    mlp1 = k.Reshape([1, -1])(mlp1)

    lstm = k.LSTM(128)(mlp1)
    mlp2 = k.Dense(50)(lstm)

    model = Model(inputs=[cnn_in, info_input], outputs=mlp2)
    return model