import matplotlib.pyplot as plt
import pandas
import torch
import numpy

if __name__ == '__main__':
    EXPERIMENT_NAME = 1674647866

    # Stati data:
    EPOCHS = 200
    BATCH_SIZE = 1000
    K_epochs = 16
    n_workers = 12

    # LOSS PLOT
    losses = pandas.read_csv(f"logs/{EXPERIMENT_NAME}_-1.csv")
    # a_actor_loss,a_critic_loss,p_a_loss,p_c_loss

    plt.plot(losses['a_actor_loss'].to_numpy(), label="Total Loss")
    plt.plot(losses['a_critic_loss'].to_numpy(), label="Critic Loss")

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(
        f"Loss in {EPOCHS} steps with {BATCH_SIZE} batch_size, k_epochs: {K_epochs}"
    )
    plt.legend()
    plt.gcf().set_size_inches(10, 7)
    plt.savefig(f"plot/{EXPERIMENT_NAME}_losses.png", dpi=400)
    plt.close()

    # Rewards
    rewards = [pandas.read_csv(f"logs/{EXPERIMENT_NAME}_{id}.csv") for id in range(n_workers)]
    rewards = sum(rewards)

    plt.plot(rewards['a'], label="Agents reward", color="blue")
    plt.plot(rewards['p'], label="Planner reward", color="red")
    plt.xlabel("Steps")
    plt.ylabel("Batch Reward")
    plt.title(
        f"Rewards in {EPOCHS} steps with {BATCH_SIZE} batch_size, k_epochs: {K_epochs}"
    )
    plt.legend()
    plt.gcf().set_size_inches(10, 7)
    plt.savefig(f"plot/{EXPERIMENT_NAME}_rewards.png", dpi=400)
    plt.close()
