"""
Multi-agent, multi-policy management algorithm.

It manages:
- multi-worker training
- batching
- giving actions to the environment
- each policy singularly so that it has all the required (correct) inputs
"""
import copy
import sys
from typing import Any, Dict, Tuple
from torch.multiprocessing import Pipe, Process

from trainer.algorithm.rollout_worker import RolloutWorker
from trainer.policies import Policy
from trainer.utils import RolloutBuffer, exec_time


def run_rollout_worker(conn, worker: RolloutWorker, id:int):
    # tmp = Memory(None, [], {}, 1, "cpu")
    while True:
        weights = conn.recv()
        # print(f"{id} presa connessione - set weights")
        worker.set_weights(weights=weights)
        # print(f"{id} batcha")
        worker.batch()
        # print(f"{id} manda memoria")
        conn.send(worker.memory)


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
    ):
        self.policies_config = policies_config
        self.train_batch_size = train_batch_size
        self.rollout_fragment_length = rollout_fragment_length
        self.num_rollout_workers = num_rollout_workers
        self.actor_keys = env.reset().keys()
        self.policy_keys = policies_config.keys()
        rollout_batch_len = self.train_batch_size // self.num_rollout_workers

        # Spawn main rollout worker, used for (actual) learning
        self.main_rollout_worker = RolloutWorker(
            rollout_fragment_length=rollout_fragment_length,
            batch_iterations = 0,
            policies_config=self.policies_config,
            actor_keys=self.actor_keys,
            policy_mapping_function=self.policy_mapping_function,
            env=env,
            device=device,
        )

        self.memory = {}
        for key in self.policy_keys:
            self.memory[key] = RolloutBuffer()

        ### Multi-processing
        # Spawn secondary workers used for batching
        self.pipes = [Pipe() for _ in range(self.num_rollout_workers)]

        # Calculate batch iterations distribution
        if self.train_batch_size % self.rollout_fragment_length != 0:
            ValueError(f"train_batch_size % rollout_fragment_length must be == 0")

        batch_iterations = self.train_batch_size//self.rollout_fragment_length
        iterations_per_worker = batch_iterations//self.num_rollout_workers
        remaining_iterations = batch_iterations-(iterations_per_worker*self.num_rollout_workers)

        self.workers = []
        for id in range(self.num_rollout_workers):
            # Get pipe connection
            parent_conn, child_conn = self.pipes[id]

            # Calculate worker_iterations
            if remaining_iterations > 0:
                worker_iterations = iterations_per_worker + 1
                remaining_iterations-=1
            else:
                worker_iterations = iterations_per_worker

            worker = RolloutWorker(
                rollout_fragment_length=rollout_fragment_length,
                batch_iterations = worker_iterations,
                policies_config=self.policies_config,
                actor_keys=self.actor_keys,
                policy_mapping_function=self.policy_mapping_function,
                env=env,
                device=device,
                id=id
            )

            p = Process(
                target=run_rollout_worker,
                name=f"RolloutWorker-{id}",
                args=(child_conn, worker, id),
            )

            self.workers.append(p)
            p.start()
            parent_conn.send(self.main_rollout_worker.get_weights())

        print("Rollout workers build!")

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
    def train_one_step(self, env):
        """
        Train all policies.
        It creates a batch of size = `self.batch_size`, then
        this RolloutBuffer is splitted between each policy following
        `self.policy_mapping_fun` and trained respectivly to the
        corrisponding policy.

        Args:
            env: updated environment
        """
        # TODO: improve distribution algo
        # batch_size_counter = 0
        # while batch_size_counter < self.train_batch_size:
        #     print(f"batch: {batch_size_counter}")
        #     # memories = [pipe[0].recv() for pipe in self.pipes]
        #     memories = []
        #     for pipe in self.pipes:
        #         memories.append(pipe[0].recv())

        #     batch_size_counter += (
        #         self.rollout_fragment_length * self.num_rollout_workers
        #     )

        #     for memory in memories:
        #         for key in self.policy_keys:
        #             # print(len(memory[key].actions))
        #             self.memory[key].extend(memory[key])

        #     if batch_size_counter != self.train_batch_size - 1:
        #         for pipe in self.pipes:
        #             pipe[0].send(1)

        # Get batches and create a single "big" batch
        memories = [pipe[0].recv() for pipe in self.pipes]
        for memory in memories:
                for key in self.policy_keys:
                    # print(len(memory[key].actions))
                    self.memory[key].extend(memory[key])


        print(f"MEMORY LEN: {len(self.memory['p'].actions)}")
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
            # worker.join()
            worker.terminate()
            # worker.close()

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

    def sync_weights(self) -> None:
        """
        Sync weights on multiple workers.
        """
        # TODO: check if deepcopy is faster or set/get weighes is better.
        # weights = self.main_rollout_worker.get_weights()

        # set weights on other workers:
        # self.second_rollout_worker.set_weights(weights=weights)
        for worker in self.workers:
            worker.policies = copy.deepcopy(self.main_rollout_worker.policies)
