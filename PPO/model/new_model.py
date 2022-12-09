import numpy as np
import tensorflow as tf
import keras.backend as K
from model.new_model_config import ModelConfig

WORLD_MAP = "world-map"
WORLD_IDX_MAP = "world-idx_map"
ACTION_MASK = "action_mask"


def apply_logit_mask(logits, mask):
    """
    Mask values of 1 are valid actions.
    Add huge negative values to logits with 0 mask values.
    """
    logit_mask = tf.ones_like(logits) * -10000000
    logit_mask = logit_mask * (1 - mask)

    return logits + logit_mask


class Model:
    """
    Actor-Critic (Policy and Value Function) Model.

    Input:
        obs, seq_in, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v
        
    Output:
        logits, values, state_h_p, state_c_p, state_h_v, state_c_v

    Brief model explanation:
    TODO

    Args:
    modelConfig: class.ModelConfig
    """

    def __init__(self, modelConfig: ModelConfig) -> None:
        """
        Initialize the Actor (Policy) Model.
        """
        self.modelConfig = modelConfig
        ### Initialize some variables needed here
        self.cell_size = modelConfig.cell_size
        self.num_outputs = modelConfig.action_space
        input_dict = {}
        non_convolutional_input_list = []
        # logits, values, state_h_p, state_c_p, state_h_v, state_c_v = (None,None,None,None,None,None,)
        ###

        ### This is for managing all the possible inputs without having more networks
        for key, value in self.modelConfig.observation_space.items():
            if isinstance(value, np.ndarray):
                # Original
                shape = (None,) + value.shape
                # shape = value.shape
            else:
                # Original
                shape = (None,) + (len(value),)
                # shape = (len(value),)
            input_dict[key] = tf.keras.layers.Input(shape=shape, name=key)

            ### Check if the input must go through a Convolutional Layer
            if key == ACTION_MASK:
                pass
            elif key == WORLD_MAP:
                conv_shape_r, conv_shape_c, conv_map_channels = (
                    value.shape[1],
                    value.shape[2],
                    value.shape[0],
                )
            elif key == WORLD_IDX_MAP:
                conv_idx_channels = value.shape[0] * self.modelConfig.emb_dim
            else:
                non_convolutional_input_list.append(key)
        ###

        ### Convolutional Layers
        conv_shape = (
            conv_shape_r,
            conv_shape_c,
            conv_map_channels + conv_idx_channels,
        )
        conv_input_map = tf.keras.layers.Permute((1, 3, 4, 2))(input_dict[WORLD_MAP])
        conv_input_idx = tf.keras.layers.Permute((1, 3, 4, 2))(
            input_dict[WORLD_IDX_MAP]
        )
        ###

        ### Concatenate all the non convolutional inputs
        non_convolutional_input = tf.keras.layers.concatenate(
            [input_dict[key] for key in non_convolutional_input_list]
        )  # , axis=-1)
        ###

        ### Define the cell states and the hidden states
        state_in_h_p = tf.keras.layers.Input(shape=(self.cell_size,), name="h_pol")
        state_in_c_p = tf.keras.layers.Input(shape=(self.cell_size,), name="c_pol")
        state_in_h_v = tf.keras.layers.Input(shape=(self.cell_size,), name="h_val")
        state_in_c_v = tf.keras.layers.Input(shape=(self.cell_size,), name="c_val")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in")
        ###

        ### Define the policy and value function models
        ### Two possibilities: '_policy' and '_value'
        ### The '_policy' model is used to sample actions
        ### The '_value' model is used to estimate the value of a state
        for tag in ["_policy", "_value"]:
            ### Define the embedding layer that embedd from 100 to 4
            map_embedding = tf.keras.layers.Embedding(
                input_dim=self.modelConfig.input_emb_vocab,
                output_dim=self.modelConfig.emb_dim,
                name="map_embedding" + tag,
            )
            ###
            ### Reshape world-idx_map from (*,11,11,2) to (2,11,11,8)
            ### FIXME
            conv_idx_embedding = tf.keras.layers.Reshape(
                (-1, conv_shape_r, conv_shape_c, conv_idx_channels)
            )(map_embedding(conv_input_idx))
            ###
            ### Concatenate the convolutional inputs
            conv_input = tf.keras.layers.concatenate(
                [conv_input_map, conv_idx_embedding], axis=-1
            )
            ###
            ### Define the convolutional module
            conv_module = tf.keras.models.Sequential(name="conv_module" + tag)
            ###
            ### Add the convolutional layers
            conv_module.add(
                tf.keras.layers.Conv2D(
                    filters=self.modelConfig.filter[0],
                    kernel_size=self.modelConfig.kernel_size,
                    strides=self.modelConfig.strides,
                    activation="relu",
                    input_shape=conv_shape,
                    name="conv2D_0" + tag,
                )
            )
            for i in range(self.modelConfig.num_conv - 1):
                conv_module.add(
                    tf.keras.layers.Conv2D(
                        filters=self.modelConfig.filter[1],
                        kernel_size=self.modelConfig.kernel_size,
                        strides=self.modelConfig.strides,
                        activation="relu",
                        name=f"conv2D_{i+1}{tag}",
                    )
                )
            ###
            ### Flatten the output of the convolutional module
            conv_module.add(tf.keras.layers.Flatten(name="flatten" + tag))
            ###
            ### Apply a layer to every temporal slice of an input
            conv_td = tf.keras.layers.TimeDistributed(conv_module)(conv_input)
            ###
            ### Concatenate the convolutional and non convolutional inputs
            dense = tf.keras.layers.concatenate([conv_td, non_convolutional_input])
            ###
            ### Add FC layers to the LSTM
            for layer in range(self.modelConfig.num_fc):
                new_layer = tf.keras.layers.Dense(
                    self.modelConfig.fc_dim, activation="relu", name=f"fc_{layer}{tag}"
                )
                dense = new_layer(dense)
            ###
            ### Define the LSTM states based on the tag
            if tag == "_policy":
                state_in = [state_in_h_p, state_in_c_p]
            else:  # "_value"
                state_in = [state_in_h_v, state_in_c_v]
            ###
            ### Normalize the output of the FC layers
            dense = tf.keras.layers.LayerNormalization(name="layer_norm" + tag)(dense)
            ###
            ### Define the LSTM output and the LSTM states
            lstm_out, state_h, state_c = tf.keras.layers.LSTM(
                self.modelConfig.cell_size,
                return_sequences=True,
                return_state=True,
                name="lstm" + tag,
            )(inputs=dense, mask=tf.sequence_mask(seq_in), initial_state=state_in)
            ###
            ### Project LSTM output to logits or value
            output = tf.keras.layers.Dense(
                self.modelConfig.action_space if tag == "_policy" else 1,
                activation=tf.keras.activations.linear,
                name="logits" if tag == "_policy" else "value",
            )(lstm_out)
            ###
            ### Define the output based on the tag
            if tag == "_policy":
                state_h_p, state_c_p = state_h, state_c
                logits = apply_logit_mask(output, input_dict[ACTION_MASK])
            else:  # "_value"
                state_h_v, state_c_v = state_h, state_c
                values = output
            ###
        ### Define the model
        self.model = tf.keras.Model(
            inputs=[layer for layer in input_dict.values()]
            + [seq_in, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v],
            outputs=[logits, values, state_h_p, state_c_p, state_h_v, state_c_v],
        )
        ###

        # if log_level == logging.DEBUG:
        self.model.summary()

    def __call__(
        self, inputs, seq_in, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v
    ):
        """
        Default model's call (used instead of predict).
        TODO: Docs
        Args:
            (inputs, seq_in, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v)

            inputs: environment observation for one action
            seq_in: 1d tensor of 
            state_in_h_p: LSTM stuff
            state_in_c_p: LSTM stuff
            state_in_h_v: LSTM stuff
            state_in_c_v: LSTM stuff
        
        Returns:
            [logits, values, state_h_p, state_c_p, state_h_v, state_c_v]
            logits: action probabilities
            values: value function predictions
            state_h_p: LSTM stuff - policy (actor)
            state_c_p: LSTM stuff - policy (actor)
            state_h_v: LSTM stuff - value function (critic)
            state_c_v: LSTM stuff - value function (critic)
        """
        input = [
            tf.keras.backend.expand_dims(
                tf.keras.backend.expand_dims(inputs["world-map"], axis=0), axis=0
            ),
            tf.keras.backend.expand_dims(
                tf.keras.backend.expand_dims(inputs["world-idx_map"], axis=0), axis=0
            ),
            tf.keras.backend.expand_dims(
                tf.keras.backend.expand_dims(inputs["flat"], axis=0), axis=0
            ),
            tf.keras.backend.expand_dims(
                tf.keras.backend.expand_dims(inputs["time"], axis=0), axis=0
            ),
            tf.keras.backend.expand_dims(
                tf.keras.backend.expand_dims(inputs["action_mask"], axis=0), axis=0
            ),
            tf.keras.backend.expand_dims(
                tf.keras.backend.expand_dims(seq_in, axis=0), axis=0
            ),
            tf.convert_to_tensor(state_in_h_p),
            tf.convert_to_tensor(state_in_c_p),
            tf.convert_to_tensor(state_in_h_v),
            tf.convert_to_tensor(state_in_c_v),
        ]
        return self.model(input)

    def initial_state(self):
        """
        Get LSTM initial null state
        """
        return [
            np.zeros((1, self.cell_size), np.float32),
            np.zeros((1, self.cell_size), np.float32),
            np.zeros((1, self.cell_size), np.float32),
            np.zeros((1, self.cell_size), np.float32),
        ]

    def ppo_loss(self, y_true, y_pred):
        """
        FIXME
        Defined in https://arxiv.org/abs/1707.06347
        """
        advantages, prediction_picks, actions = (
            y_true[:, :1],
            y_true[:, 1 : 1 + self.modelConfig.action_space],
            y_true[:, 1 + self.modelConfig.action_space :],
        )
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = tf.keras.backend.clip(prob, 1e-10, 1.0)
        old_prob = tf.keras.backend.clip(old_prob, 1e-10, 1.0)

        ratio = tf.keras.backend.exp(
            tf.keras.backend.log(prob) - tf.keras.backend.log(old_prob)
        )

        p1 = ratio * advantages
        p2 = (
            tf.keras.backend.clip(
                ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING
            )
            * advantages
        )

        actor_loss = -tf.keras.backend.mean(tf.keras.backend.minimum(p1, p2))

        entropy = -(y_pred * tf.keras.backend.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * tf.keras.backend.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss

    # def add_time_dimension(self, tensor: tf.Tensor, seq_lens: int):
    #     """Adds a time dimension to padded inputs.

    #     Arguments:
    #         padded_inputs (Tensor): a padded batch of sequences. That is,
    #             for seq_lens=[1, 2, 2], then inputs=[A, *, B, B, C, C], where
    #             A, B, C are sequence elements and * denotes padding.
    #         seq_lens (Tensor): the sequence lengths within the input batch,
    #             suitable for passing to tf.nn.dynamic_rnn().

    #     Returns:
    #         Reshaped tensor of shape [NUM_SEQUENCES, MAX_SEQ_LEN, ...].
    #     """
    #     padded_batch_size = tensor.shape[0]
    #     max_seq_len = padded_batch_size // seq_lens

    #     tensor = tf.reshape(tensor, [seq_lens, max_seq_len] + list(tensor.shape[1:]))
    #     return tensor

    # def add_time_dimension(self, tensor: tf.Tensor, seq_lens: tf.Tensor):
    #     """Adds a time dimension to padded inputs.

    #     Arguments:
    #         padded_inputs (Tensor): a padded batch of sequences. That is,
    #             for seq_lens=[1, 2, 2], then inputs=[A, *, B, B, C, C], where
    #             A, B, C are sequence elements and * denotes padding.
    #         seq_lens (Tensor): the sequence lengths within the input batch,
    #             suitable for passing to tf.nn.dynamic_rnn().

    #     Returns:
    #         Reshaped tensor of shape [NUM_SEQUENCES, MAX_SEQ_LEN, ...].
    #     """
    #     padded_batch_size = tensor.shape[0] # First dimension of the tensor
    #     max_seq_len = padded_batch_size // seq_lens.shape[0] #

    #     # Dynamically reshape the padded batch to introduce a time dimension.
    #     new_batch_size = padded_batch_size // max_seq_len
    #     new_shape = ([new_batch_size, max_seq_len] +
    #                 tensor.get_shape().as_list()[1:])
    #     return tf.reshape(tensor, new_shape)

    # def _extract_input_list(self, dictionary):
    #     return [dictionary[k] for k in self._input_keys]

    # def forward(self, input_dict, state, seq_lens):
    #     """Adds time dimension to batch before sending inputs to forward_rnn().

    #     You should implement forward_rnn() in your subclass."""
    #     output, new_state = self.forward_rnn(
    #         [
    #             add_time_dimension(t, seq_lens)
    #             for t in self._extract_input_list(input_dict["obs"])
    #         ],
    #         state,
    #         seq_lens,
    #     )
    #     return tf.reshape(output, [-1, self.num_outputs]), new_state

    # def forward_rnn(self, inputs, state, seq_lens):
    #     model_out, self._value_out, h_p, c_p, h_v, c_v = self.rnn_model(
    #         inputs + [seq_lens] + state
    #     )
    #     return model_out, [h_p, c_p, h_v, c_v]

    # def get_initial_state(self):
    #     return [
    #         np.zeros(self.cell_size, np.float32),
    #         np.zeros(self.cell_size, np.float32),
    #         np.zeros(self.cell_size, np.float32),
    #         np.zeros(self.cell_size, np.float32),
    #     ]

    # def value_function(self):
    #     return tf.reshape(self._value_out, [-1])
