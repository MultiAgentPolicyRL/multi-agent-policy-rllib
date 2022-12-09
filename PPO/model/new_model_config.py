"""
Set up and data check of model's config
"""


from typing import Tuple


class ModelConfig:
    """
    a
    """

    def __init__(
        self,
        observation_space,
        action_space,
        emb_dim: int,
        cell_size: int,
        input_emb_vocab: int,
        num_conv: int,
        fc_dim: int,
        num_fc: int,
        filtering: Tuple[int, int],
        kernel_size: Tuple[int, int],
        strides: int,
    ):
        """
        Builds model config.

        Args:
            observation_space: observation space of the environment (for a single agent)
            action_space: action space of the agent selected
            emb_dim: output embedding dimension
            cell_size: size of the LSTM cell
            input_emb_vocab: input embedding dimension
            num_conv: number of convolutional layers
            fc_dim: dimension of the fully-connected layer
            num_fc: number of fully-connected layers
            filtering:
            kernel_size:
            strides:

        TODO: complete docs
        """
        self.observation_space = observation_space
        self.action_space = action_space

        self.emb_dim = emb_dim
        self.cell_size = cell_size
        self.input_emb_vocab = input_emb_vocab
        self.num_conv = num_conv
        self.fc_dim = fc_dim
        self.num_fc = num_fc
        self.filter = filtering
        self.kernel_size = kernel_size
        self.strides = strides
