#!/usr/bin/env python
# https://raw.githubusercontent.com/ray-project/ray/releases/0.8.4/rllib/examples/serving/cartpole_client.py
"""
In two separate shells run:
    $ python aie_server.py --run=[PPO|DQN]
    $ python aie_client.py --inference-mode=local|remote
"""

import argparse
import os
import gym
from ray.rllib.env.policy_client import PolicyClient
import yaml

from env_wrapper import RLlibEnvWrapper

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=bool, default=True, help="Whether to train or not.")
    parser.add_argument(
        "--inference-mode", type=str, required=True, choices=["local", "remote"])
    parser.add_argument(
        "--off-policy",
        action="store_true",
        help="Whether to take random instead of on-policy actions.")
    parser.add_argument("--run-dir", type=str,
                        help="Path to the directory for this run.", default="phase1",)
    
    args = parser.parse_args()
    config_path = os.path.join(args.run_dir, "config.yaml")
    assert os.path.isdir(args.run_dir)
    assert os.path.isfile(config_path)

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)
    
    return args, run_configuration
    

if __name__ == "__main__":
    args, run_config = process_args()
    
    trainer_config = run_config.get("trainer")
    env_config = {
        "env_config_dict": run_config.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }

    env = RLlibEnvWrapper(env_config)
        
    client = PolicyClient(
        "http://localhost:9900", inference_mode=args.inference_mode)

    eid = client.start_episode(training_enabled=args.train)
    obs = env.reset()
    rewards = 0

    # while True:
    #     # env.render()
    #     action = client.get_action(eid, obs)
    #     obs, reward, done, info = env.step(action)
    #     rewards += reward
    #     client.log_returns(eid, reward, info=info)

        # if done:
        #     print("Total reward:", rewards)
            
        #     rewards = 0
        #     client.end_episode(eid, obs)
        #     obs = env.reset()
        #     eid = client.start_episode(training_enabled=not args.no_train)
    
