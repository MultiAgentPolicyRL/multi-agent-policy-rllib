"""
Experiment launcher
"""
import time
import logging
import sys

import torch

from algorithm import Algorithm
from environment import get_environment
from policies import EmptyPolicy, PpoPolicy
from tqdm import tqdm

if __name__ == "__main__":
    EXPERIMENT_NAME = int(time.time())

    EPOCHS = 1
    BATCH_SIZE = 400
    SEED = 1
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
        'a': {
            'policy': PpoPolicy,
            'observation_space': env.observation_space,
            'action_space': [50],
            'K_epochs': K_epochs,
            'device': device,
        },
        'p': {
            'policy': EmptyPolicy,
            'observation_space': env.observation_space_pl,
            'action_space': [22, 22, 22, 22, 22, 22, 22],
        }
    }

    algorithm: Algorithm = Algorithm(
        train_batch_size=BATCH_SIZE,
        policies_config=policies,
        env=env,
        device=device,
        num_rollout_workers=1,
        rollout_fragment_length=200,
    )

    for i in range(EPOCHS):
        actions, _ = algorithm.get_actions(obs)
        obs, rew, done, info = env.step(actions)
        algorithm.train_one_step(env=env)
        # algorithm.compare_models(obs)
