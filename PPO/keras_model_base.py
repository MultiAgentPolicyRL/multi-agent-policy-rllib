import tensorflow as tf
import keras.layers as k
from keras import Model
import numpy as np

def feed_model(obs, model):
    """ Takes in input an observation (for a single agent (e.g., obs['0'])) and a model and returns the output. """
    # numerical_features = []

    # for data in obs.keys():
    #     numerical_features.append(obs[data])

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
    #return obs['action_mask'] * model([obs['world-map'], obs['flat']])
    return tf.reshape(model([obs['world-map'], obs['flat']]), [-1])


def get_policy_model(conv_filters, filter_size):
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
    #output = k.Dense(1, activation='relu')(mlp2)

    model = Model(inputs=[cnn_in, info_input], outputs=mlp2)

    return model

def get_model(state_shape, action_dim, units=(400, 300, 100)):
    state = k.Input(shape=state_shape)

    # Value function (baseline)
    # used to calculate advantage estimate
    vf = k.Dense(units[0], name="Value_L0", activation="tanh")(state)
    for index in range(1, len(units)):
        vf = k.Dense(units[index], name="Value_L{}".format(
            index), activation="tanh")(vf)

    value_pred = k.Dense(1, name="Out_value")(vf)

    # Our Policy
    cnn_in = k.Input(shape=(7, 11, 11))
    map_cnn = k.Conv2D(state_shape[0], action_dim, activation='relu')(cnn_in)
    map_cnn = k.Conv2D(state_shape[1], action_dim, activation='relu')(map_cnn)
    map_cnn = k.Flatten()(map_cnn)

    info_input = k.Input(shape=(136))
    mlp1 = k.Concatenate()([map_cnn, info_input])
    mlp1 = k.Dense(128, activation='relu')(mlp1)
    mlp1 = k.Dense(128, activation='relu')(mlp1)
    mlp1 = k.Reshape([1, -1])(mlp1)

    lstm = k.LSTM(128)(mlp1)
    mlp2 = k.Dense(action_dim, activation="softmax")(lstm)

    #pi = k.Dense(units[0], name="Policy_L0", activation="tanh")(state)
    #for index in range(1, len(units)):
    #    pi = k.Dense(units[index], name="Policy_L{}".format(
    #        index), activation="tanh")(pi)
    #
    #action_probs = k.Dense(action_dim, name="Out_probs",
    #                     activation='softmax')(pi)
    model = Model(inputs=state, outputs=[mlp2, value_pred])

    model.summary()

    return model

#if __name__ == '__main__':
#model = ctach_model((136,), 55)
#print(model([np.random.rand(1, 136)]))
# model.compile(optimizer='adam', loss='mse')
# model.fit([np.random.rand(1, 136), np.random.rand(1, 136)], [np.random.rand(1, 50), np.random.rand(1, 1)], epochs=1)
