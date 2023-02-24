"""
Experiment launcher
"""
import time

import torch
from trainer.algorithm import Algorithm
from trainer.environment import get_environment
from trainer.policies import EmptyPolicy, PpoPolicy

from tqdm import tqdm

if __name__ == "__main__":

    EXPERIMENT_NAME = int(time.time())
    print(f"EXPERIMENT_NAME: {EXPERIMENT_NAME}")
    # batch_size 1600, 4 workers, rollout_lenghts 200 -> ~11s/step
    # batch_size 1600, 4 workers, rollout_lenghts 200 -> ~5,6s/step - torch.tensor: 1.32
    # batch_size 1600, 4 workers, rollout_lenghts 200 -> ~5,6s/step - torch.from_numpy + np.array: 1.62
    # With files instead of pipes:
    # batch_size 1600, 4 workers, rollout_lenghts 200 -> ~2.2s/step
    # batch_size 1600, 4 workers, rollout_lenghts 200 -> ~1.6s/step
    # 24/1/23 with files in disk: Function train_one_step Took 6.888880795999967 seconds - 6k batch - 12 workers - 200
    # 25/1/23 with files in disk: Function train_one_step Took 4.844272016000104 seconds - 6k batch - 12 workers - 200
    # 14:55
    
    EPOCHS = 1
    BATCH_SIZE = 200
    SEED = 1
    NUM_WORKERS = 1
    ROLLOUT_FRAGMENT_LENGTH = 200
    K_EPOCHS = 16
    PLOTTING = True

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cuda'
    DEVICE = "cpu"

    env = get_environment(DEVICE)
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
        device=DEVICE,
        num_rollout_workers=NUM_WORKERS,
        rollout_fragment_length=ROLLOUT_FRAGMENT_LENGTH,
        experiment_name=EXPERIMENT_NAME,
        seed=SEED,
    )

    for i in tqdm(range(EPOCHS)):
    # for i in range(EPOCHS):

        actions, _ = algorithm.get_actions(obs)
        obs, rew, done, info = env.step(actions)
        algorithm.train_one_step()

    algorithm.close_workers()
