from typing import Tuple

"""
Set up and data check of model's config
"""


class ModelConfig:
    """
    a
    """

    def __init__(
        self,
        action_space,
        observation_space,
        name: str,
        emb_dim: int = 4,
        cell_size: int = 128,
        input_emb_vocab: int = 100,
        num_conv: int = 2,
        fc_dim: int = 128,
        num_fc: int = 2,
        filter: Tuple[int, int] = (16, 32),
        kernel_size: Tuple[int, int] = (3, 3),
        strides: int = 2,
        output_size: int = 50,
        lr: float = 0.001,  # was 0.0003
        device: str = "cpu",
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.name = name
        self.emb_dim = emb_dim
        self.cell_size = cell_size
        self.input_emb_vocab = input_emb_vocab
        self.num_conv = num_conv
        self.fc_dim = fc_dim
        self.num_fc = num_fc
        self.filter = filter
        self.kernel_size = kernel_size
        self.strides = strides
        self.output_size = output_size
        self.lr = lr
        self.device = device
