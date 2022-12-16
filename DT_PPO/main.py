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
    algorithm = PPO(env, 50, batch_size=1000, log_level=logging.INFO, log_path=log_path)

    iterations = 50
    total_rewards = 0
    
    for it in range(iterations):
        logger.info(f"Starting iteration {it+1}/{iterations}")

        states, actions, rewards, predictions, next_states, values, total_rewards = algorithm.populate_batch(total_rewards=total_rewards) # Torch
        # states, actions, rewards, predictions, next_states, values, _ = algorithm.populate_batch() # Tensorflow

        # for agent in ['0', '1', '2', '3', 'p']:
        #     # rewards is a dict of agents
        #     # rerwards[agent] is a list of tensor rewards for each agent
        #     temp_rewards = [x.squeeze(0).item() for x in rewards[agent]]
        #     logger.info(f"Agent {agent} rewards | mean: {round(np.mean(temp_rewards), 3)}, std: {round(np.std(temp_rewards), 3)}, min: {round(np.min(temp_rewards), 3)}, max: {round(np.max(temp_rewards), 3)}")
        logger.info(f"Total rewards: {total_rewards}")

        losses = algorithm.train(states, actions, rewards, predictions, next_states, values,) # Torch
        # losses = algorithm.train(states, actions, rewards, predictions, next_states, values, _) # Tensorflow
        
        # logger.info(f"Losses: {losses}")
        logger.info(f"{'#'*50}")
        
    exit()