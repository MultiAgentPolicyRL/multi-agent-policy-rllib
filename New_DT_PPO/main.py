import os
import sys
import random
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

### Helpful for type declarations and logs
from ai_economist.foundation.base.base_env import BaseEnvironment

###
from ai_economist_ppo_dt.utils import create_environment, get_basic_logger
from ai_economist_ppo_dt.torch import PPO

# from ai_economist_ppo_dt.tensorflow import Actor, Critic, PPO

os.environ["CUDA_VISIBLE_DEVICES"] = (
    "-1"
    if not tf.config.list_physical_devices("GPU")
    else len(tf.config.list_physical_devices("GPU"))
)
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
    env.reset()

    # Init PPO
    algorithm = PPO(env, 50, batch_size=1000, log_level=logging.INFO, log_path=log_path)
    iterations = 50

    logger.warning(
        f"The 'p' agent is not considered for now! In future it must be fixed."
    )

    for it in range(iterations):
        logger.info(f"Starting iteration {it+1}/{iterations}")

        states, actions, rewards, predictions, next_states = algorithm.populate_batch()

        for agent in ["0", "1", "2", "3"]:
            logger.info(
                f"Agent {agent} rewards | mean: {round(np.mean(rewards[agent]), 3)}, std: {round(np.std(rewards[agent]), 3)}, min: {round(np.min(rewards[agent]), 3)}, max: {round(np.max(rewards[agent]), 3)}"
            )

        losses = algorithm.train(states, actions, rewards, predictions, next_states)

        # rearrage losses array:
        actor_loss = []
        critic_loss = []
        for agent in ["0", "1", "2", "3"]:
            actor_loss.append(round(losses[agent]["actor"][-1], 3))
            critic_loss.append(round(losses[agent]["critic"][-1], 3))
        logger.info(f"Iteration: {it+1}, Agent losses: {actor_loss}")
        logger.info(f"Iteration: {it+1}, Critic losses: {critic_loss}")

        logger.info(f"{'#'*50}")

        # Test
        # rewards = algorithm.test()

        # for agent in ['0', '1', '2', '3']:
        #     logger.info(f"Agent {agent} rewards | mean: {round(np.mean(rewards[agent]), 3)}, std: {round(np.std(rewards[agent]), 3)}, min: {round(np.min(rewards[agent]), 3)}, max: {round(np.max(rewards[agent]), 3)}")
