import os
import sys
import torch
import random
import logging
import numpy as np
from datetime import datetime

### Helpful for type declarations and logs
from ai_economist.foundation.base.base_env import BaseEnvironment

###
from ai_economist_ppo_dt.utils import create_environment, get_basic_logger
from ai_economist_ppo_dt.torch import PPO

# from ai_economist_ppo_dt.tensorflow import PPO
# import tensorflow as tf

# Check if tensorflow is imported
if "tf" in sys.modules:
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        "-1"
        if not tf.config.list_physical_devices("GPU")
        else len(tf.config.list_physical_devices("GPU"))
    )
    device = (
        tf.config.list_physical_devices("GPU")[0]
        if tf.config.list_physical_devices("GPU")
        else "cpu"
    )

if "torch" in sys.modules:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
SEED = random.randrange(min(sys.maxsize, 2 ** 32 - 1))

if __name__ == "__main__":
    # To avoid warnings from bad ai-ecoomist code
    os.system("clear")

    # Configure logger
    log_path = os.path.join(
        os.getcwd(), "logs", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = get_basic_logger("main", log_path=log_path)

    # Create the environment
    env: BaseEnvironment = create_environment()
    env.seed(SEED)
    state = env.reset()

    ## DEBUG ONLY
    buffer_size = 1000
    log_level = logging.INFO
    gettrace = getattr(sys, "gettrace", None)
    if gettrace is not None and gettrace():
        buffer_size = 6
        log_level = logging.DEBUG

    # Init PPO
    algorithm = PPO(
        env,
        action_space=50,
        buffer_size=buffer_size,
        epochs=10,
        log_level=log_level,
        log_path=log_path,
    )

    iterations = 50
    total_rewards = 0
    rewards_list = []
    temp = {"Total": [], "Actor": [], "Critic": []}
    losses_list = {
        "0": temp,
        "1": temp,
        "2": temp,
        "3": temp,
        "p": temp,
    }

    try:
        for it in range(iterations):
            logger.info(f"{'#'*50}")
            logger.info(
                f"Starting iteration {it+1}/{iterations} | buffer_size: {algorithm.buffer_size} | total_rewards: {round(total_rewards, 2)}"
            )

            # Populate batch
            (
                states,
                actions,
                rewards,
                predictions,
                next_states,
                values,
                total_rewards,
            ) = algorithm.populate_batch(
                total_rewards=total_rewards
            )  # Torch
            # states, actions, rewards, predictions, next_states, values, _ = algorithm.populate_batch() # Tensorflow

            # Save total rewards
            rewards_list.append(round(total_rewards, 3))

            # Train
            losses = algorithm.train(
                states,
                actions,
                rewards,
                predictions,
                next_states,
                values,
            )  # Torch
            # losses = algorithm.train(states, actions, rewards, predictions, next_states, values, _) # Tensorflow

            # Save losses
            for key in losses.keys():
                losses_list[key]["Total"].append(losses[key]["Total"])
                losses_list[key]["Actor"].append(losses[key]["Action"])
                losses_list[key]["Critic"].append(losses[key]["Value"])

            # logger.info(f"Losses: {losses}")

        # Plot rewards
        import matplotlib.pyplot as plt

        plt.plot(rewards_list)
        plt.savefig("rewards.png")
        plt.cla()

        # Plot losses with color based on 'Total', 'Actor', 'Critic'
        # For each agent plot different plots (4) that are all in the same figure

        # import plotly.express as px
        # import pandas as pd
        # for key in losses_list.keys():
        #     df = pd.DataFrame(losses_list[key])
        #     df = df.reset_index()
        #     df = df.rename(columns={'index': 'Iteration'})
        #     fig = px.line(df, x="Iteration", y="Total", color='Critic')
        #     fig.write_image(f"losses_{key}.png")

    except KeyboardInterrupt:
        logger.info(f"{'#'*50}")
        logger.info(f"KeyboardInterrupt")
        logger.info(f"{'#'*50}")

    exit()
