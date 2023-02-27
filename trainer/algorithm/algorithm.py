"""
Multi-agent, multi-policy management algorithm.

It manages:
- multi-worker training
- batching
- giving actions to the environment
- each policy singularly so that it has all the required (correct) inputs
"""
import pickle
import sys
from time import sleep
from typing import Dict, Tuple
from torch.multiprocessing import Pipe, Process
import ray
from trainer.algorithm.rollout_worker import RolloutWorker
from trainer.policies import Policy
from trainer.utils import RolloutBuffer, exec_time
from os import remove
import time
from copy import deepcopy


def run_rollout_worker(conn, worker: RolloutWorker, id: int):
    # tmp = Memory(None, [], {}, 1, "cpu")
    while True:
        weights = conn.recv()
        worker.set_weights(weights=weights)
        worker.batch()
        # Bad way of using semaphores/signals
        conn.send(1)
        worker.save_csv()


class Algorithm(object):
    """
    Multi-agent, multi-policy management algorithm.
    """

    def __init__(
        self,
        train_batch_size: int,
        policies_config: Dict[str, Policy],
        env,
        device: str,
        num_rollout_workers: int,
        rollout_fragment_length: int,
        experiment_name,
        seed: int,
    ):
        self.policies_config = policies_config
        self.train_batch_size = train_batch_size
        self.rollout_fragment_length = rollout_fragment_length
        self.num_rollout_workers = num_rollout_workers
        self.actor_keys = env.reset().keys()
        self.policy_keys = policies_config.keys()
        self.experiment_name = experiment_name

        # Spawn main rollout worker, used for (actual) learning
        self.main_rollout_worker : RolloutWorker = RolloutWorker.remote(
            rollout_fragment_length=rollout_fragment_length,
            batch_iterations=0,
            policies_config=self.policies_config,
            actor_keys=self.actor_keys,
            policy_mapping_function=self.policy_mapping_function,
            env=env,
            device=device,
            seed=seed,
            experiment_name=experiment_name
        )

        self.memory = {}
        for key in self.policy_keys:
            self.memory[key] = RolloutBuffer()

        # Multi-processing
        # Spawn secondary workers used for batching
        # After batching all data is merged together to create a single big
        # batch.

        # Calculate batch iterations distribution
        if self.train_batch_size % self.rollout_fragment_length != 0:
            ValueError(f"train_batch_size % rollout_fragment_length must be == 0")

        batch_iterations = self.train_batch_size // self.rollout_fragment_length
        iterations_per_worker = batch_iterations // self.num_rollout_workers
        remaining_iterations = batch_iterations - (
            iterations_per_worker * self.num_rollout_workers
        )

        self.workers = []
        for _id in range(self.num_rollout_workers):
            # Calculate worker_iterations
            if remaining_iterations > 0:
                worker_iterations = iterations_per_worker + 1
                remaining_iterations -= 1
            else:
                worker_iterations = iterations_per_worker

            self.workers.append(RolloutWorker.remote(
                rollout_fragment_length=rollout_fragment_length,
                batch_iterations=worker_iterations,
                policies_config=self.policies_config,
                actor_keys=self.actor_keys,
                policy_mapping_function=self.policy_mapping_function,
                env=env,
                device=device,
                id=_id,
                seed=seed,
                experiment_name=self.experiment_name,
            ))


        print("Rollout workers built!")

    def policy_mapping_function(self, key: str) -> str:
        """
        It differenciates between two types of agents:
        `a` and `p`:    `a` -> economic player
                        `p` -> social planner

        Args:
            key: agent dictionary key
        """
        if str(key).isdigit() or key == "a":
            return "a"
        return "p"

    @exec_time
    def train_one_step(self):
        """
        Train all policies.
        It creates a batch of size = `self.batch_size`, then
        this RolloutBuffer is splitted between each policy following
        `self.policy_mapping_fun` and trained respectivly to the
        corrisponding policy.
        """
        # FIXME: remove this deepcopy and prepare data in a more effective way
        models_weiths = ray.put(deepcopy(ray.get(self.main_rollout_worker.get_weights.remote())))
        
        for worker in self.workers:
            worker.set_weights.remote(models_weiths)

        # Get batches and create a single "big" batch
        # batch_ref = [ worker.batch.remote() for worker in self.workers]
        start = time.time()        
        results = ray.get([worker.batch.remote() for worker in self.workers])
        print(f"{time.time() - start}")
        # for rollout in results:
        #     for key in self.policy_keys:
        #         self.memory[key].extend(rollout[key])
        # start = time.time()        
        # mem = ray.put(results)

        # Update main worker policy
        self.main_rollout_worker.learn(memory=self.memory)

        # Send updated policy to all rollout workers
        for pipe in self.pipes:
            pipe[0].send(self.main_rollout_worker.get_weights())

        # Clear memory from used batch
        for memory in self.memory.values():
            memory.clear()

    def close_workers(self):
        """
        Kill and clear all workers.
        """
        for worker in self.workers:
            worker.kill()

    def get_actions(self, obs: dict) -> Tuple[dict, dict]:
        """
        Build action dictionary using actions taken from all policies.

        Args:
            obs: environment observation

        Returns:
            policy_action
            policy_logprob
        """
        policy_action, policy_logprob = ray.get(self.main_rollout_worker.get_actions.remote(obs=obs))

        return policy_action, policy_logprob

    def delete_data_files(self):
        """
        Delte communication files used during batching
        """
        for file_name in range(self.num_rollout_workers):
            remove(f"/tmp/{self.experiment_name}_{file_name}.bin")
