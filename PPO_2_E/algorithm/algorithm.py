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
from algorithm.rollout_worker import RolloutWorker
import multiprocessing as mp

from policies import Policy
from utils import exec_time, RolloutBuffer


def run_rollout_worker(conn, worker: RolloutWorker):
    # tmp = Memory(None, [], {}, 1, "cpu")
    while True:
        # FIXME: aggiungere if vai avanti a fare batch / aggiorna modello.
        # print("waiting for weights")
        policies = conn.recv()
        # print("updating policies")
        worker.policies = policies
        # worker.policies = copy.deepcopy(policies)
        # print("batching")
        worker.batch()
        # print("data to queue")
        conn.send(worker.memory)
        # conn.send(tmp)
        # print("banana")


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

        # Spawn main rollout worker, used for (actual) learning
        self.main_rollout_worker = RolloutWorker(
            rollout_fragment_length=rollout_fragment_length,
            policies_config=self.policies_config,
            actor_keys=self.actor_keys,
            policy_mapping_function=self.policy_mapping_function,
            env=env,
            device=device,
        )

        self.memory = {}
        for key in self.policy_keys:
            self.memory[key] = RolloutBuffer()

        # Multi-processing
        # Spawn secondary workers used for batching
        self.pipes = [mp.Pipe() for _ in range(self.num_rollout_workers)]

        self.workers = []
        for id in range(self.num_rollout_workers):
            parent_conn, child_conn = self.pipes[id]

            worker = RolloutWorker(
                rollout_fragment_length=rollout_fragment_length,
                policies_config=self.policies_config,
                actor_keys=self.actor_keys,
                policy_mapping_function=self.policy_mapping_function,
                env=env,
                device=device,
            )

            p = mp.Process(
                target=run_rollout_worker,
                name=f"RolloutWorker-{id}",
                args=(child_conn, worker),
            )

            self.workers.append(p)
            p.start()
            parent_conn.send(self.main_rollout_worker.policies)

        print("done")

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
        print("one step")
        # TODO: improve distribution algo
        batch_size_counter = 0
        while batch_size_counter < self.train_batch_size:
            print(f"batch: {batch_size_counter}")
            memories = [pipe[0].recv() for pipe in self.pipes]
            batch_size_counter += (
                self.rollout_fragment_length * self.num_rollout_workers
            )

            for memory in memories:
                for key in self.policy_keys:
                    self.memory[key].extend(memory[key])
        
        print(f"MEMORY LEN: {len(self.memory['a'].actions)}")
        self.main_rollout_worker.learn(memory=self.memory)

        for pipe in self.pipes:
            pipe[0].send(self.main_rollout_worker.policies)

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

    def compare_models(self, obs):
        # self.second_rollout_worker.policies = copy.deepcopy(self.main_rollout_worker.policies)

        policy_action1, policy_logprob1 = self.main_rollout_worker.get_actions(obs)
        print(policy_action1, policy_logprob1)
        policy_action1, policy_logprob1 = self.main_rollout_worker.get_actions(obs)
        print(policy_action1, policy_logprob1)
        policy_action1, policy_logprob1 = self.main_rollout_worker.get_actions(obs)
        print(policy_action1, policy_logprob1)

        policy_action2, policy_logprob2 = self.second_rollout_worker.get_actions(obs)
        print(policy_action2, policy_logprob2)
        policy_action2, policy_logprob2 = self.second_rollout_worker.get_actions(obs)
        print(policy_action2, policy_logprob2)
        policy_action2, policy_logprob2 = self.second_rollout_worker.get_actions(obs)
        print(policy_action2, policy_logprob2)

        print(self.main_rollout_worker.get_weights())
        print(
            self.second_rollout_worker.get_weights()
            == self.main_rollout_worker.get_weights()
        )
        a1 = self.main_rollout_worker.get_weights()
        b1 = self.second_rollout_worker.get_weights()
        # print(a1['a']['a']-b1['a']['a'])

        # check actor weights
        for a, b in zip(a1["a"]["a"], b1["a"]["a"]):
            print(a == b)
        # check critic weights
        for a, b in zip(a1["a"]["c"], b1["a"]["c"]):
            print(a == b)
        # check optim weights
        for a, b in zip(a1["a"]["o"], b1["a"]["o"]):
            print(a == b)
