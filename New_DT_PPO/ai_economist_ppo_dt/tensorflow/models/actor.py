import os
import logging
import numpy as np
import tensorflow as tf
import keras.layers as k
import keras.backend as K
from typing import Optional

####
from keras.optimizers import Adam
from keras.models import Model, load_model

from ai_economist_ppo_dt.utils import get_basic_logger, time_it


class Actor:
    """
    Actor (Policy) Model.
    =====




    """

    def __init__(
        self,
        action_space: int = 50,
        conv_filters: tuple = (16, 32),
        filter_size: int = 3,
        log_level: int = logging.INFO,
        log_path: str = None,
    ) -> None:
        """
        Actor (Policy) Model.
        -----
        Parameters
        ----------
        """
        self.action_space = action_space
        self.logger = get_basic_logger("Actor", level=log_level, log_path=log_path)

        # Model
        self.device = "/cpu:0" if "CUDA_VISIBLE_DEVICES" in os.environ else "/gpu:0"
        self.model = self._build_model(action_space, conv_filters, filter_size)

    def _build_model(
        self, action_space: int, conv_filters: tuple, filter_size: int
    ) -> Model:
        """
        Build an actor (policy) network that maps states (for now only world_map and flat) -> actions

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
            The actor model.
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
            lstm_out = k.Dense(action_space, activation="sigmoid")(lstm)
            # lstm_out = k.Dense(action_space, activation='sigmoid')(concat)

            # Model --- Learning rate = 0.0003 because of https://github.com/ray-project/ray/issues/8091
            model = Model(inputs=[cnn_in, info_input], outputs=lstm_out)
            model.compile(
                loss=self._loss_wrapper(action_space),
                optimizer=Adam(learning_rate=0.0003),
                run_eagerly=False,
            )

        # Log
        self.logger.debug("Actor model summary:")
        if self.logger.level == logging.DEBUG:
            model.summary()

        # Return model
        return model

    def _loss_wrapper(self, num_actions: int) -> tf.Tensor:
        """
        Wrapper for the loss function.

        Parameters
        ----------
        num_actions : int
            The number of actions.

        Returns
        -------
        loss : tf.Tensor
            The loss function.
        """

        @tf.function
        def loss(y_true, y_pred) -> tf.Tensor:
            """
            Loss function for the actor (policy) model.
            -----
            Defined in [arxiv:1707.06347](https://arxiv.org/abs/1707.06347).

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

                advantages, prediction_picks, actions = (
                    y_true[:, :1],
                    y_true[:, 1 : 1 + num_actions],
                    y_true[:, -1:],
                )

                LOSS_CLIPPING = 0.2
                ENTROPY_LOSS = 0.001

                prob = actions * y_pred
                old_prob = actions * prediction_picks

                prob = K.clip(prob, 1e-10, 1.0)
                old_prob = K.clip(old_prob, 1e-10, 1.0)

                ratio = K.exp(K.log(prob) - K.log(old_prob))

                p1 = ratio * advantages
                p2 = (
                    K.clip(
                        ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING
                    )
                    * advantages
                )

                actor_loss = -K.mean(K.minimum(p1, p2))

                entropy = -(y_pred * K.log(y_pred + 1e-10))
                entropy = ENTROPY_LOSS * K.mean(entropy)

                total_loss = actor_loss - entropy

            return total_loss

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
        Predict an action given a `input_state`.

        Parameters
        ----------
        input_state : np.ndarray or list
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
        #     return tf.squeeze(self.model(input_state))
        #
        # return self.model.predict(input_state, verbose=0, steps=len(input_state), workers=workers, use_multiprocessing=use_multiprocessing)

        # New
        with tf.device(self.device):
            actions = tf.squeeze(self.model(input_state))

        return tf.divide(actions, tf.reduce_sum(actions))

    # @tf.function
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        batch_size: int = 1,
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
        self.model.save(os.path.join(path, "actor.h5"))

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
        self.model: Model = load_model(
            os.path.join(path, "actor.h5"),
            custom_objects={"loss": self._loss_wrapper(self.action_space)},
        )
        self.model.compile(
            loss=self._loss_wrapper(self.action_space),
            optimizer=Adam(learning_rate=0.0003),
        )
