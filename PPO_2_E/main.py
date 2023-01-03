"""
Experiment launcher
"""
import datetime
import logging
import sys
from environment.env import get_environment
from utils.setup_logger import setup_logger
from policies.empty_policy import EmptyPolicy
from algorithm.algorithm import Algorithm
if __name__ == "__main__":
    EXPERIMENT_NAME = datetime.datetime.now()

    env = get_environment()

    EPOCHS = 10
    SEED = 1

    env = get_environment()
    env.seed(SEED)
    obs = env.reset()

    policies = {
        'a': EmptyPolicy(observation_space=env.observation_space, action_space=[50], batch_size=0),
        'p': EmptyPolicy(observation_space=env.observation_space_pl, action_space=[22,22,22,22,22,22,22], batch_size=0)
    }

    algorithm : Algorithm = Algorithm(1, policies=policies, env=env)

    for i in range(2):
        actions, _ = algorithm.get_actions(obs)
        obs, rew, done, info = env.step(actions)
        algorithm.train_one_step()

    # env.close()
