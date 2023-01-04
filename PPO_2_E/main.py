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

    EPOCHS = 5
    BATCH_SIZE = 6000
    SEED = 1
    K_epochs = 8
    plotting = True

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cuda'
    device = 'cpu'
    print(device)

    env = get_environment(device)
    env.seed(SEED)
    torch.manual_seed(SEED)
    obs = env.reset()

    

    policies = {
        "a": PpoPolicy(observation_space=env.observation_space, action_space=[50], K_epochs=K_epochs, device=device),
        "p": EmptyPolicy(
            observation_space=env.observation_space_pl,
            action_space=[22, 22, 22, 22, 22, 22, 22],
        ),
    }

    algorithm: Algorithm = Algorithm(batch_size=BATCH_SIZE, policies=policies, env=env, device=device)

    returns = []
    for i in (range(EPOCHS)):
        actions, _ = algorithm.get_actions(obs)
        # print(actions)
        obs, rew, done, info = env.step(actions)
        # print(f"REW: {rew['0']+rew['1']+rew['2']+rew['3']}")
        
        losses = algorithm.train_one_step(env=env)
        returns.append(losses)
        print("\n\n")

        # print(
        #     f"A: batch_rew: {losses['a']['rew']}, a_loss: {losses['a']['actor']}, c_loss: {losses['a']['critic']}, "
        # )
    sys.exit()
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
        plt.title(f"Loss in {EPOCHS} steps with {BATCH_SIZE} batch_size, k_epochs: {K_epochs}")
        plt.legend()
        plt.gcf().set_size_inches(10, 7)
        plt.savefig(f"plot/{EXPERIMENT_NAME}_losses.png", dpi=400)
        plt.close()

        ## batch_rew
        plt.plot(a0, '--', label = "Reward Actor 0", color='mediumslateblue', alpha=0.7)
        plt.plot(a1, '--', label = "Reward Actor 1", color='chocolate', alpha=0.7)
        plt.plot(a2, '--', label = "Reward Actor 2", color='olivedrab', alpha=0.7)
        plt.plot(a3, '--', label = "Reward Actor 3", color='turquoise', alpha=0.7)

        plt.plot(a_rews, label = "Total Reward", color='blue')
        plt.xlabel("Steps")
        plt.ylabel("Batch Reward")
        plt.title(f"Rewards in {EPOCHS} steps with {BATCH_SIZE} batch_size, k_epochs: {K_epochs}")
        plt.legend()
        plt.gcf().set_size_inches(10, 7)
        plt.savefig(f"plot/{EXPERIMENT_NAME}_rewards.png", dpi=400 )
        plt.close()
