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
        emb_dim:
        cell_size:
        input_emb_vocab:
        num_conv:
        fc_dim:
        num_fc:
        filtering:
        kernel_size:
        strides:
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

        """
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
        log_level: int = logging.INFO,
        log_path: str = None,
        """
