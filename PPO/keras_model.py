import tensorflow as tf
from typing import Tuple


class AieModel():
    """
    Combined actor-critic network
    NN Model as described in the paper and in it's repo.
    """

    def __init__(self):
        # action_dimension FIXME
        self.action_dim = 1  # 1 # 50

        self.cnn_in = tf.keras.Input(shape=(7, 11, 11))
        self.map_cnn = tf.keras.layers.Conv2D(16, 3,
                                              activation='relu')(self.cnn_in)
        self.map_cnn = tf.keras.layers.Conv2D(32, 3,
                                              activation='relu')(self.map_cnn)
        self.map_cnn = tf.keras.layers.Flatten()(self.map_cnn)

        self.info_input = tf.keras.Input(shape=(136))
        self.mlp1 = tf.keras.layers.Concatenate()(
            [self.map_cnn, self.info_input])
        self.mlp1 = tf.keras.layers.Dense(128, activation='relu')(self.mlp1)
        self.mlp1 = tf.keras.layers.Dense(128, activation='relu')(self.mlp1)
        self.mlp1 = tf.keras.layers.Reshape([1, -1])(self.mlp1)

        self.lstm = tf.keras.layers.LSTM(128)(self.mlp1)

        # Value function
        self.value_pred = tf.keras.layers.Dense(1,
                                                name="Out_value_function",
                                                activation='softmax')(
                                                    self.lstm)
        # Policy pi
        self.action_probs = tf.keras.layers.Dense(self.action_dim,
                                                  name="Out_probs_actions",
                                                  activation='softmax')(
                                                      self.lstm)

        self.model = tf.keras.Model(
            inputs=[self.cnn_in, self.info_input],
            outputs=[self.action_probs, self.value_pred])

    def call(self, obs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Feed the neural network
        """
        # return tf.reshape(self.model([obs['world-map'], obs['flat']]), [-1])
        output = self.model([obs['world-map'], obs['flat']])
        return tf.reshape(output[0], [-1]), output[1]



if __name__ == '__main__':
    model : AieModel = AieModel()
    model.model.summary()
#model = ctach_model((136,), 55)
#print(model([np.random.rand(1, 136)]))
# model.compile(optimizer='adam', loss='mse')
# model.fit([np.random.rand(1, 136), np.random.rand(1, 136)], [np.random.rand(1, 50), np.random.rand(1, 1)], epochs=1)
