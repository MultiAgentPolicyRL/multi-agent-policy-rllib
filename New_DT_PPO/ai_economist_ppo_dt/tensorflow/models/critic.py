import os
import logging
from typing import Optional
import numpy as np
import tensorflow as tf
import keras.layers as k
import keras.backend as K

####
from keras.optimizers import Adam
from keras.models import Model, load_model

from ai_economist_ppo_dt.utils import get_basic_logger


class Critic:
    """
    Actor (Policy) Model.
    =====




    """

    def __init__(
        self,
        conv_filters: tuple = (16, 32),
        filter_size: int = 3,
        log_level: int = logging.INFO,
        log_path: str = None,
    ) -> None:
        """
        Critic (Policy) Model.
        -----
        Parameters
        ----------
        """
        self.logger = get_basic_logger("Critic", level=log_level, log_path=log_path)

        # Model
        self.device = "/cpu:0" if "CUDA_VISIBLE_DEVICES" in os.environ else "/gpu:0"
        self.model = self._build_model(conv_filters, filter_size)

    def _build_model(self, conv_filters: tuple, filter_size: int) -> Model:
        """
        Build an critic (policy) network that maps states (for now only world_map and flat) -> actions

        Parameters
        ----------
        action_space : int
            Dimension of the action space.
        conv_filters : tuple
            Number of filters for each convolutional layer.
        filter_size : int
            Size of the convolutional filters.

        Returns
        -------
        model : Model
            The critic model.
        """
        with tf.device(self.device):
            # Input layers
            cnn_in = k.Input(shape=(7, 11, 11))
            info_input = k.Input(shape=(136,))

            # CNN
            cnn = k.Conv2D(conv_filters[0], filter_size, activation="relu")(cnn_in)
            cnn = k.Conv2D(conv_filters[1], filter_size, activation="relu")(cnn)
            cnn = k.Flatten()(cnn)

            # Concatenate CNN and info_input
            concat = k.concatenate([cnn, info_input])
            concat = k.Dense(128, activation="relu")(concat)
            concat = k.Dense(128, activation="relu")(concat)
            concat = k.Reshape([1, -1])(concat)

            # LSTM
            lstm = k.LSTM(128, unroll=True)(concat)

            # Output
            lstm_out = k.Dense(1, activation=None)(lstm)

            # Model
            model = Model(inputs=[cnn_in, info_input], outputs=lstm_out)
            # model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(learning_rate=0.0003), run_eagerly=False)
            # custom loss
            model.compile(
                loss=self._loss_wrapper(),
                optimizer=Adam(learning_rate=0.0003),
                run_eagerly=False,
            )

            # Log
            self.logger.debug("Critic model summary:")
            if self.logger.level == logging.DEBUG:
                model.summary()

            # Return model
            return model

    def _loss_wrapper(self, flag_mean: bool = True) -> tf.Tensor:
        """
        Wrapper for the loss function.

        Parameters
        ----------
        flag_mean : bool
            Whether to return the mean loss or the clipped loss.

        Returns
        -------
        loss : tf.Tensor
            The loss function.
        """

        @tf.function
        def loss(
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
        ) -> tf.Tensor:
            """
            -----

            Parameters
            -----
            y_true : tf.Tensor
                The true action.
            y_pred : tf.Tensor
                The predicted action.

            Returns
            -----
            loss : tf.Tensor
                The loss.
            """
            with tf.device(self.device):
                y_true = tf.squeeze(y_true)

                values = K.expand_dims(y_true[1], -1)
                y_true = K.expand_dims(y_true[0], -1)

                if flag_mean:
                    return K.mean((y_true - y_pred) ** 2)

                # Keras mean_squared_error
                # return K.mean(K.square(y_true - y_pred), axis=-1)

                # L_CLIP
                LOSS_CLIPPING = 0.2
                clipped_value_loss = values + K.clip(
                    y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING
                )
                v_loss1 = (y_true - clipped_value_loss) ** 2
                v_loss2 = (y_true - y_pred) ** 2
                value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))

            return value_loss

        return loss

    @tf.function
    def predict(
        self,
        input_state: np.ndarray,
        verbose: Optional[int] = 0,
        workers: Optional[int] = 8,
        use_multiprocessing: Optional[bool] = True,
    ) -> np.ndarray:
        """
        Predict an action given a input_state.

        Parameters
        ----------
        input_state : np.ndarray
            The input_state of the environment.

        Returns
        -------
        action : np.ndarray
            The action to take.
        """
        # # Deprecated
        # # if run_eagerly is enabled in model compile,
        # # to speed up the prediction
        # if self.model.run_eagerly:
        #     return np.squeeze(self.model(input_state).numpy())

        # return self.model.predict(input_state, verbose=0, steps=len(input_state), workers=workers, use_multiprocessing=use_multiprocessing)

        # New
        with tf.device(self.device):
            return tf.squeeze(self.model(input_state))

    # @tf.function
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        batch_size: int = 32,
        verbose: bool = False,
        shuffle: bool = True,
        workers: int = 8,
        use_multiprocessing: bool = True,
    ) -> None:
        """
        Fit the model.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        y : np.ndarray
            The target data.
        epochs : int (default: 1)
            The number of epochs.
        batch_size : int (default: 50)
            The batch size.
        verbose : bool (default: False)
            Verbosity mode.
        shuffle : bool
            Whether to shuffle the data.
        workerks : int (default: 8)
            The number of workers.
        use_multiprocessing : bool (default: True)
            Whether to use multiprocessing. If `True`, `workers` must be a positive integer.

        Returns
        -------
        History : np.ndarray
            The history of the model losses.
        """
        if use_multiprocessing and workers <= 0:
            raise ValueError(
                "If `use_multiprocessing` is `True`, `workers` must be a positive integer."
            )

        with tf.device(self.device):
            return self.model.fit(
                x,
                y,
                epochs=epochs,
                batch_size=batch_size,
                steps_per_epoch=batch_size // epochs,
                verbose=verbose,
                shuffle=shuffle,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
            )

    def save_weights(self, path: str) -> None:
        """
        Save the model weights.

        Parameters
        ----------
        path : str
            The path to save the weights.
        """
        self.model.save_weights(path)

    def save(self, path: str, loss: int) -> None:
        """
        Save the model.

        Parameters
        ----------
        path : str
            The path to save the model.
        """
        self.model.save(os.path.join(path, "critic.h5"))

    def load_weights(self, path: str) -> None:
        """
        Load the model weights.

        Parameters
        ----------
        path : str
            The path to load the weights.
        """
        self.model.load_weights(path)

    def load(self, path: str) -> None:
        """
        Load the model.

        Parameters
        ----------
        path : str
            The path to load the model.
        """
        model: Model = load_model(
            os.path.join(path, "critic.h5"),
            custom_objects={"loss": self._loss_wrapper(self.action_space)},
        )
        model.compile(
            loss=self._loss_wrapper(self.action_space),
            optimizer=Adam(learning_rate=0.0003),
        )

        self.model = model
