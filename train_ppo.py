"""
Experiment launcher
"""
import time
import logging
import sys

import torch
from trainer.algorithm import Algorithm
from trainer.environment import get_environment
from trainer.policies import EmptyPolicy, PpoPolicy

from tqdm import tqdm

torch.multiprocessing.set_start_method("fork")

if __name__ == "__main__":
    EXPERIMENT_NAME = int(time.time())
    print(f"EXPERIMENT_NAME: {EXPERIMENT_NAME}")

    EPOCHS = 200
    BATCH_SIZE = 6000
    SEED = 1

    NUM_WORKERS = 12
    ROLLOUT_FRAGMENT_LENGTH = 200
    K_EPOCHS = 16

    PLOTTING = True

    DEVICE = "cpu"

    """ To switch between phase 1 and phase 2 load saved models
    (with that specific experiment name) and change 'p' policy's
    to PpoPolicy and set it's required parameters.
    """

    LOAD_SAVED_MODELS = 0

    env = get_environment()
    env.seed(SEED)
    torch.manual_seed(SEED)
    obs = env.reset()

    policies = {
        "a": {
            "policy": PpoPolicy,
            "observation_space": env.observation_space,
            "action_space": 50,
            "K_epochs": K_EPOCHS,
            "device": DEVICE,
        },
        "p": {
            "policy": EmptyPolicy,
            "observation_space": env.observation_space_pl,
            "action_space": [22, 22, 22, 22, 22, 22, 22],
        },
    }

    algorithm: Algorithm = Algorithm(
        train_batch_size=BATCH_SIZE,
        policies_config=policies,
        env=env,
        num_rollout_workers=NUM_WORKERS,
        rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH,
        experiment_name=EXPERIMENT_NAME,
        seed=SEED,
        load_saved_models=LOAD_SAVED_MODELS,
    )

    del env

    for i in tqdm(range(EPOCHS)):
        # actions, _ = algorithm.get_actions(obs)
        # obs, rew, done, info = env.step(actions)
        algorithm.train_one_step()

    algorithm.save_models()
    algorithm.close_workers()
