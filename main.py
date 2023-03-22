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
import logging
import multiprocessing
from src.common import get_environment

from src import PpoTrainConfig, DtTrainConfig, InteractConfig

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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(process)d-%(levelname)s-%(message)s"
    )

    env = get_environment()

    trainer = PpoTrainConfig(
        get_mapping_function,
        env,
        num_workers=10,
        step=1,
        batch_size=4000,
        rollout_fragment_length=200,
        mapped_agents={"a": True, "p": True},
    )
    trainer.train()

    # trainer = DtTrainConfig(
    #     env,
    #     episodes=2,
    #     episode_len=100,
    #     lambda_=2,
    #     generations=2,
    #     mapped_agents={"a": True, "p": True},
    # )
    # trainer.train()

    # interact = InteractConfig(get_mapping_function, env, PpoTrainConfig, config={}, mapped_agents={
    #     "a": "PPO_P1_22-03-2023_1679498536_1",
    #     "p": False,
    # })


    # interact = InteractConfig(get_mapping_function, env, DtTrainConfig, config={}, mapped_agents={
    #     "a": "DT_P2_2023-03-22_163328_2",
    #     "p": False,
    # })
