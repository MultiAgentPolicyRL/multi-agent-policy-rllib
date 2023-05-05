"""
This file describes the config of an experiment.
An experiment can do training or can load a pre
trained model(s) and interact with the environment.

Available training algorithms are about two different
families:
- online learning
    - DT_ql
- offline learning
    - PPO
"""
import argparse
import logging
import multiprocessing
from src.common import get_environment

from src import PpoTrainConfig, DtTrainConfig, InteractConfig
from src.train.ppo_dt import PPODtTrainConfig

# pylint: disable=pointless-string-statement

# Configuration declaration
"""
Select how the manager should work: training or interaction with
pre-trained model(s) -> mode = 'train' or 'interaction'.
Then select the type of algorithm you want to use, so you can select
online learning: 'online', or offline learning: 'offline'. Select how many
steps of training will be done.
After that there are specific parameters that needs to be setup.
Seed.

If you are doing 'interaction' mode you have to set the 'path' of
models weights'.

Interaction:
    type
    path -> mapped
    mapping_function
    model's directory id (path)

Training:
    Online learning params:
        mapping_function
    Offline learning params:
        k_epochs
        eps_clip
        gamma
        device
        learning_rate
        num_workers
        mapping_function
        agents_to_train (list)
"""


def get_mapping_function():
    """
    Returns a mapping function
    """

    def mapping_function(key: str) -> str:
        """
        It differenciates between two types of agents:
        `a` and `p`:    `a` -> economic player
                        `p` -> social planner

        Args:
            key: agent dictionary key
        """
        if str(key).isdigit() or key == "a":
            return "a"
        return "p"

    return mapping_function

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train',
                    help='Mode of the experiment')
parser.add_argument('--type', type=str, default='PPO_DT',
                    help='Type of the algorithm')
parser.add_argument('--path-ppo', type=str, default=None,
                    help='Path of the model weights for ppo')
parser.add_argument('--path-dt', type=str, default=None,
                    help='Path of the model weights for dt')

args = parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(process)d - %(asctime)s - %(levelname)s - %(message)s"
    )

    env = get_environment()

    if args.mode == "train":
        if args.type == "PPO":
            trainer = PpoTrainConfig(
                get_mapping_function,
                env,
                num_workers=12,
                step=1000,
                batch_size=6000,
                rollout_fragment_length=200,
                mapped_agents={
                    "a": True, 
                    "p": True
                },
            )
            trainer.train()
        elif args.type == "DT":
            trainer = DtTrainConfig(
                env,
                episodes=5,
                episode_len=1000,
                lambda_=180,
                generations=50,
                mapped_agents={
                    "a": True, 
                    "p": True
                },
            )
            trainer.train()
        elif args.type == "PPO_DT":
            trainer = PPODtTrainConfig(
                env,
                episodes=6,
                episode_len=1000,
                lambda_=50,
                generations=30,
                mapped_agents={
                    "a": 'PPO_P1_01-04-2023_1680328509_1000', # This must be the folder name to load the agent pre-trained in pytorch
                    "p": True
                },
            )
            trainer.train()
        else:
            raise ValueError("Invalid type of algorithm")
        
    elif args.mode == "eval":
        if args.type == "PPO":
            interact = InteractConfig(get_mapping_function, env, PpoTrainConfig, config={}, mapped_agents={
                "a": "PPO_P1_01-04-2023_1680328509_1000",
                "p": False,
            })
        elif args.type == "DT":
            interact = InteractConfig(get_mapping_function, env, DtTrainConfig, config={}, mapped_agents={
                "a": "DT_P2_2023-04-19_124132_5",
                "p": "DT_P2_2023-04-19_124132_5",
            })
        elif args.type == "PPO_DT":
            interact = InteractConfig(get_mapping_function, env, PPODtTrainConfig, config={}, mapped_agents={
                "a": "PPO_DT_P2_2023-04-17_181557_10",
                "p": "PPO_DT_P2_2023-04-17_181557_10",
            })
    else:
        raise ValueError("Invalid mode")
    
    # logging.info("Done")