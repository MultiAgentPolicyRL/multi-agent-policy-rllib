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
from utils import exec_time, Memory
from utils import Memory

def run_rollout_worker(queue, worker: RolloutWorker):
    while True:
        policies = queue.get()
        worker.policies = copy.deepcopy(policies)
        worker.batch()
        queue.put(worker.memory)

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

        obs: Dict[str, Any] = env.reset()

        self.policies_size = {}
        available_agent_id = []
        
        for key in policies_config.keys():
            self.policies_size[key] = 0
        
        for key in obs.keys():
            self.policies_size[self.policy_mapping_function(key)] += 1
            available_agent_id.append(key)

        # Spawn main rollout worker, used for (actual) learning
        self.main_rollout_worker = RolloutWorker(
            rollout_fragment_length=rollout_fragment_length,
            policies_config=self.policies_config,
            available_agent_id=available_agent_id,
            policies_size=self.policies_size,
            policy_mapping_function=self.policy_mapping_function,
            env=env,
            device=device,
        )

        self.memory = Memory(
            policy_mapping_fun=self.policy_mapping_function,
            available_agent_id=available_agent_id,
            batch_size=self.train_batch_size,
            policy_size=self.policies_size,
            device=device,
        )

        # Multi-processing
        # Spawn secondary workers used for batching
        # self.workers = []
        # for _ in range(self.num_rollout_workers):
        #     self.workers.append(RolloutWorker(
        #         rollout_fragment_length=rollout_fragment_length,
        #         policies_config=self.policies_config,
        #         available_agent_id=available_agent_id,
        #         policies_size=self.policies_size,
        #         policy_mapping_function=self.policy_mapping_function,
        #         env=env,
        #         device=device,
        #     ))
        self.queue = [mp.Queue() for _ in range(self.num_rollout_workers)]
        workers = [
            RolloutWorker(
                rollout_fragment_length=rollout_fragment_length,
                policies_config=self.policies_config,
                available_agent_id=available_agent_id,
                policies_size=self.policies_size,
                policy_mapping_function=self.policy_mapping_function,
                env=env,
                device=device,
            )
            for _ in range(self.num_rollout_workers)
        ]

        # with mp.Pool() as pool:
        with mp.Pool(initializer=run_rollout_worker, initargs=(self.queue, workers)) as pool:
            pool.map(run_rollout_worker, [(queue, worker) for worker,queue in zip(workers, self.queue)])

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
        # self.sync_weights()
        # self.batch()
        
        # self.main_rollout_worker.learn(memory=self.memory)
        # TODO: improve distribution algo
        batch_size_counter = 0
        while batch_size_counter < self.train_batch_size:
            memories = [queue.get() for queue in self.queue]
            batch_size_counter += self.rollout_fragment_length*self.num_rollout_workers

            for memory in memories:
                self.memory.append(memory)
            
        self.main_rollout_worker.learn(memory=self.memory)
        for queue in self.queue:
            queue.put(self.main_rollout_worker.policies)



    # @exec_time
    def batch(self):
        """
        Distributed batching.
        Creates a batch of `self.train_batch_size` steps saved in `self.rollout_buffer`.
        """
        # FIXME: not final form
        batch_size_counter = 0
        while batch_size_counter < self.train_batch_size:
            for worker in self.workers:
                batch_size_counter+=self.rollout_fragment_length
                worker.batch()
                self.memory.append(worker.memory)

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
        print(self.second_rollout_worker.get_weights() == self.main_rollout_worker.get_weights())
        a1=self.main_rollout_worker.get_weights()
        b1=self.second_rollout_worker.get_weights()
        # print(a1['a']['a']-b1['a']['a'])
        
        # check actor weights
        for a,b in zip(a1['a']['a'], b1['a']['a']):
            print(a==b)
        # check critic weights
        for a,b in zip(a1['a']['c'], b1['a']['c']):
            print(a==b)
        # check optim weights
        for a,b in zip(a1['a']['o'], b1['a']['o']):
            print(a==b)
