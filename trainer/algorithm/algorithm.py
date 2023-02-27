"""
Multi-agent, multi-policy management algorithm.

It manages:
- multi-worker training
- batching
- giving actions to the environment
- each policy singularly so that it has all the required (correct) inputs
"""
import pickle
from time import sleep
from typing import Dict, Tuple
from torch.multiprocessing import Pipe, Process

from trainer.algorithm.rollout_worker import RolloutWorker
from trainer.policies import Policy
from trainer.utils import RolloutBuffer, exec_time
from os import remove
import time


def run_rollout_worker(conn, worker: RolloutWorker, id: int):
    # tmp = Memory(None, [], {}, 1, "cpu")
    while True:
        weights = conn.recv()
        print(f"{id} - got weights")
        worker.set_weights(weights=weights)
        print(f"{id} - set w")
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
        self.main_rollout_worker = RolloutWorker(
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
        self.pipes = [Pipe() for _ in range(self.num_rollout_workers)]

        # TODO: add shared memory - probably one per process w/the main process
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
        self.workers_id = []
        for _id in range(self.num_rollout_workers):
            # Get pipe connection
            parent_conn, child_conn = self.pipes[_id]

            # Calculate worker_iterations
            if remaining_iterations > 0:
                worker_iterations = iterations_per_worker + 1
                remaining_iterations -= 1
            else:
                worker_iterations = iterations_per_worker

            worker = RolloutWorker(
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
            )

            self.workers_id.append(_id)

            p = Process(
                target=run_rollout_worker,
                name=f"RolloutWorker-{_id}",
                args=(child_conn, worker, _id),
            )

            self.workers.append(p)
            p.start()
            parent_conn.send(self.main_rollout_worker.get_weights())

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

    # @exec_time
    def train_one_step(self):
        """
        Train all policies.
        It creates a batch of size = `self.batch_size`, then
        this RolloutBuffer is splitted between each policy following
        `self.policy_mapping_fun` and trained respectivly to the
        corrisponding policy.
        """
        # Get batches and create a single "big" batch
        # Bad way to do semaphores
        print("bad semaphores")
        _ = [pipe[0].recv() for pipe in self.pipes]

        # Open all files in a list of `file`
        # _time = time.perf_counter()
        for file_name in self.workers_id:
            file = open(f"/tmp/{self.experiment_name}_{file_name}.bin", "rb")
            rollout = pickle.load(file)

            for key in self.policy_keys:
                self.memory[key].extend(rollout[key])

            file.close()

        # print(f"UNPICKLING TIME: {time.perf_counter()-_time}")

        # Update main worker policy
        self.main_rollout_worker.learn(memory=self.memory)
        print("learned!")
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
        policy_action, policy_logprob = self.main_rollout_worker.get_actions(obs=obs)

        return policy_action, policy_logprob

    def delete_data_files(self):
        """
        Delte communication files used during batching
        """
        for file_name in range(self.num_rollout_workers):
            remove(f"/tmp/{self.experiment_name}_{file_name}.bin")
