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

if __name__ == "__main__":
    EXPERIMENT_NAME = int(time.time())
    # batch_size 1600, 4 workers, rollout_lenghts 200 -> ~11s/step
    # batch_size 1600, 4 workers, rollout_lenghts 200 -> ~5,6s/step - torch.tensor: 1.32
    # batch_size 1600, 4 workers, rollout_lenghts 200 -> ~5,6s/step - torch.from_numpy + np.array: 1.62
    EPOCHS = 5
    BATCH_SIZE = 3000
    SEED = 1

    NUM_WORKERS = 12
    rollout_fragment_length = 500

    K_epochs = 8
    plotting = True

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cuda'
    device = "cpu"

    env = get_environment(device)
    env.seed(SEED)
    torch.manual_seed(SEED)
    obs = env.reset()

    # FIXME: mode policy creation in rollout_worker -> so it can be "distributed"
    policies = {
        "a": {
            "policy": PpoPolicy,
            "observation_space": env.observation_space,
            "action_space": [50],
            "K_epochs": K_epochs,
            "device": device,
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
        device=device,
        num_rollout_workers=NUM_WORKERS,
        rollout_fragment_length=rollout_fragment_length,
    )

    for i in range(EPOCHS):
        actions, _ = algorithm.get_actions(obs)
        obs, rew, done, info = env.step(actions)
        algorithm.train_one_step(env=env)
        # algorithm.compare_models(obs)

    algorithm.close_workers()
