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
if 'tf' in sys.modules:
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" if not tf.config.list_physical_devices('GPU') else len(tf.config.list_physical_devices('GPU'))
    device = tf.config.list_physical_devices('GPU')[0] if tf.config.list_physical_devices('GPU') else 'cpu'

if 'torch' in sys.modules:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
SEED = random.randrange(min(sys.maxsize, 2**32-1))

if __name__ == "__main__":   
    # To avoid warnings from bad ai-ecoomist code
    os.system('clear') 

    # Configure logger
    log_path = os.path.join(os.getcwd(), 'logs', f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_basic_logger("main", log_path=log_path)

    # Create the environment
    env:BaseEnvironment = create_environment()
    env.seed(SEED)
    state = env.reset()  
    
    # Init PPO
    algorithm = PPO(env, action_space=50, batch_size=1000, epochs=30, log_level=logging.INFO, log_path=log_path)

    iterations = 10
    total_rewards = 0
    rewards_list = []
    losses_list = {'0': [], '1': [], '2': [], '3': [], 'p': [],}
    
    try:
        for it in range(iterations):
            logger.info(f"{'#'*50}")
            logger.info(f"Starting iteration {it+1}/{iterations} | batch_size: {algorithm.batch_size} | total_rewards: {round(total_rewards, 2)}")

            # Populate batch
            states, actions, rewards, predictions, next_states, values, total_rewards = algorithm.populate_batch(total_rewards=total_rewards) # Torch
            # states, actions, rewards, predictions, next_states, values, _ = algorithm.populate_batch() # Tensorflow

            # Save total rewards
            rewards_list.append(total_rewards)

            # Train
            losses = algorithm.train(states, actions, rewards, predictions, next_states, values,) # Torch
            # losses = algorithm.train(states, actions, rewards, predictions, next_states, values, _) # Tensorflow

            # Save losses
            for key in losses.keys():
                losses_list[key].append(losses[key][-1])
            
            # logger.info(f"Losses: {losses}")
        logger.info(f"{'#'*50}")

        # Plot rewards
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.lineplot(x=range(len(rewards_list)), y=rewards_list)
        plt.show()

        # Plot losses from losses_list and color by key
        for key in losses_list.keys():
            sns.lineplot(x=range(len(losses_list[key])), y=losses_list[key], label=key)
        plt.show()
    except KeyboardInterrupt:
        logger.info(f"{'#'*50}")
        logger.info(f"KeyboardInterrupt")
        logger.info(f"{'#'*50}")
        
    
    exit()