"""
Experiment launcher
"""
import datetime
import logging
import sys

import torch

from algorithm import Algorithm
from environment import get_environment
from policies import EmptyPolicy, PpoPolicy
from utils import setup_logger
from tqdm import tqdm
if __name__ == "__main__":
    EXPERIMENT_NAME = datetime.datetime.now()

    env = get_environment()

    EPOCHS = 400
    BATCH_SIZE = 1000
    SEED = 1
    plotting = True

    env = get_environment()
    env.seed(SEED)
    torch.manual_seed(SEED)
    obs = env.reset()

    policies = {
        "a": PpoPolicy(observation_space=env.observation_space, action_space=[50]),
        "p": EmptyPolicy(
            observation_space=env.observation_space_pl,
            action_space=[22, 22, 22, 22, 22, 22, 22],
        ),
    }

    algorithm: Algorithm = Algorithm(batch_size=BATCH_SIZE, policies=policies, env=env)

    returns = []
    for i in tqdm(range(EPOCHS)):
        actions, _ = algorithm.get_actions(obs)
        # print(actions)
        obs, rew, done, info = env.step(actions)
        # print(f"REW: {rew['0']+rew['1']+rew['2']+rew['3']}")
        
        losses = algorithm.train_one_step(env=env)
        
        returns.append(losses)
        print(
            f"A: batch_rew: {losses['a']['rew']}, a_loss: {losses['a']['actor']}, c_loss: {losses['a']['critic']}, "
        )

    # Plotting
    if plotting:
        # Plotting only AGENTS
        import matplotlib.pyplot as plt
        # Extract data
        a_a_loss, a_c_loss, a_rews = [], [], []
        a0,a1,a2,a3 = [],[],[],[]
        for item in returns:
            a_a_loss.append(item['a']['actor'])
            a_c_loss.append(item['a']['critic'])
            a_rews.append(item['a']['rew'])
            a0.append(item['a']['a0'])
            a1.append(item['a']['a1'])
            a2.append(item['a']['a2'])
            a3.append(item['a']['a3'])


        ## a_loss, c_loss        
        plt.plot(a_a_loss, label = "Actor Loss")
        plt.plot(a_c_loss, label = "Critic Loss")
        # plt.plot(y_map, critic_loss, label = "Critic Loss")

        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title(f"Loss in {EPOCHS} steps with {BATCH_SIZE} batch_size")
        plt.legend()
        plt.savefig("actor_loss.png")
        plt.close()

        ## batch_rew
        plt.plot(a_rews, label = "Total Reward")
        plt.plot(a0, label = "Reward Actor 0")
        plt.plot(a1, label = "Reward Actor 1")
        plt.plot(a2, label = "Reward Actor 2")
        plt.plot(a3, label = "Reward Actor 3")
        plt.xlabel("Steps")
        plt.ylabel("Batch Reward")
        plt.title(f"Rewards in {EPOCHS} steps with {BATCH_SIZE} batch_size")
        plt.legend()
        plt.savefig("rewards.png")
        plt.close()