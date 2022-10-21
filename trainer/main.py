import os
import sys
import warnings

import ray
from ray.tune import CLIReporter
from tqdm import tqdm

#from ai_economist_ppo.launch import build_trainer
from build_trainer import build_trainer
from utils import (get_basic_logger, process_args, save_data,
                                    set_up_dirs)

warnings.filterwarnings("ignore")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Outputs")

logger = get_basic_logger("MAIN")


def main():
    ### Process the arguments
    run_directory, run_configuration, redis_password, ip_address, cluster = process_args()
    
    ### If cluster is True, connect to the cluster, first check if ray is already running
    if ray.is_initialized():
        logger.info("Ray is already initialized. Closing it...")
        ray.shutdown()

    if cluster:
        logger.info("Connecting to cluster...\r")
        try:
            ray.init(address=ip_address, redis_password=redis_password, log_to_driver=False)
        except:
            logger.info("Failed to connect to cluster. Check your IP address and password.")
            sys.exit()
        finally:
            logger.info("Connected to cluster.")
    else:
        try:
            ray.init(log_to_driver=False)
        except:
            logger.info("Failed to initiate local cluster.")
            sys.exit()
        finally:
            logger.info("Ready to train locally.")

    ### Create trainer objects
    #agents, planner = build_trainer(run_configuration)
    agents = build_trainer()

    ### Set up directories for logging and saving. Restore if not completed
    log_directory, restore = set_up_dirs(OUTPUT_FOLDER)  
    start = 0

    if restore:
        start = 0 

    ### Create environment for training
    # env_agents = agents.workers.local_worker().env
    # env_planner = planner.workers.local_worker().env

    # env_agents.reset()
    # env_planner.reset()

    reporter = CLIReporter(max_progress_rows=10)

    ### Start training
    #bar = tqdm(range(start, run_configuration["general"]["episodes"]))
    bar = tqdm(range(start, 1))
    bar.set_description("Start training")
    for i in bar:
        ### Train agents
        new_agents = agents.train()

        ### Train planner
        #new_planner = planner.train()
        bar.set_description(f"Agents: {new_agents['episode_reward_mean']:.2f}")# Planner: {new_planner['episode_reward_mean']:.2f}")
        
        ### Swap the weights to synchronize
        #agents.set_weights(planner.get_weights(["planner_policy"]))
        #planner.set_weights(agents.get_weights(["agent_policy"]))

        ### Save
        checkpoint_agent = agents.save(os.path.join(log_directory, "agents"))
        #checkpoint_planner = planner.save(os.path.join(log_directory, "planner"))
        #reporter.report(**new_agents)
    
    agents.stop()

    ### Evaluation
    run_configuration["num_workers"] = 0
    #agents, planner = build_trainer(run_configuration)
    agents = build_trainer()

    agents.restore(checkpoint_agent)
    #planner.restore(checkpoint_planner)

    ### Create environment for evaluation
    env_agents = agents.workers.local_worker().env
    #env_planner = planner.workers.local_worker().env

    obs:dict = env_agents.reset()
    done = False
    eval_results = {"eval_reward": 0, "eval_eps_length": 0}

    while not done:
        obs.pop("p")
        for k in obs["0"].keys():
            for agent in obs.keys():
                with open("file.txt", "a") as f:
                    f.write(f"{agent} {k}: {obs[agent][k].shape} and dtype {obs[agent][k].dtype} and min {obs[agent][k].min()} and max {obs[agent][k].max()}\n")
        with open("file.txt", "a") as f:
            f.write(f"{agents.workers.local_worker().preprocessors.get('agent_policy')._obs_space}\n")
        action = agents.compute_action(obs, policy_id="agent_policy")
        obs, reward, done, info = env_agents.step(action)
        eval_results["eval_reward"] += reward
        eval_results["eval_eps_length"] += 1
    results = {**new_agents, **eval_results}

    #obs = env_planner.reset()
    agent = 'p'
    done = False
    eval_results = {"eval_reward": 0, "eval_eps_length": 0}

    while not done:
        action = agents.compute_action(obs[agent], policy_id=agent)
        obs, reward, done, info = env_agents.step(action)
        eval_results[agent]["eval_reward"] += reward
        eval_results[agent]["eval_eps_length"] += 1
    results = {**new_agents, **eval_results}
    #tune.report(results)

    ray.shutdown()

if __name__ == "__main__":
    main()
