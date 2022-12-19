# from ai_economist import foundation
# from env_wrapper import EnvWrapper
import random
import numpy as np
import torch


# env_config = {
#     "env_config_dict": {
#         # ===== SCENARIO CLASS =====
#         # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
#         # The environment object will be an instance of the Scenario class.
#         "scenario_name": "layout_from_file/simple_wood_and_stone",
#         # ===== COMPONENTS =====
#         # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
#         #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
#         #   {component_kwargs} is a dictionary of kwargs passed to the Component class
#         # The order in which components reset, step, and generate obs follows their listed order below.
#         "components": [
#             # (1) Building houses
#             (
#                 "Build",
#                 {
#                     "skill_dist": "pareto",
#                     "payment_max_skill_multiplier": 3,
#                     "build_labor": 10,
#                     "payment": 10,
#                 },
#             ),
#             # (2) Trading collectible resources
#             (
#                 "ContinuousDoubleAuction",
#                 {
#                     "max_bid_ask": 10,
#                     "order_labor": 0.25,
#                     "max_num_orders": 5,
#                     "order_duration": 50,
#                 },
#             ),
#             # (3) Movement and resource collection
#             ("Gather", {"move_labor": 1, "collect_labor": 1, "skill_dist": "pareto"}),
#             # (4) Planner
#             (
#                 "PeriodicBracketTax",
#                 {
#                     "period": 100,
#                     "bracket_spacing": "us-federal",
#                     "usd_scaling": 1000,
#                     "disable_taxes": False,
#                 },
#             ),
#         ],
#         # ===== SCENARIO CLASS ARGUMENTS =====
#         # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
#         "env_layout_file": "quadrant_25x25_20each_30clump.txt",
#         "starting_agent_coin": 10,
#         "fixed_four_skill_and_loc": True,
#         # ===== STANDARD ARGUMENTS ======
#         # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
#         "n_agents": 4,  # Number of non-planner agents (must be > 1)
#         "world_size": [25, 25],  # [Height, Width] of the env world
#         "episode_length": 2000,  # Number of timesteps per episode
#         # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
#         # Otherwise, the policy selects only 1 action.
#         "multi_action_mode_agents": False,
#         "multi_action_mode_planner": True,
#         # When flattening observations, concatenate scalar & vector observations before output.
#         # Otherwise, return observations with minimal processing.
#         "flatten_observations": True,
#         # When Flattening masks, concatenate each action subspace mask into a single array.
#         # Note: flatten_masks = True is required for masking action logits in the code below.
#         "flatten_masks": True,
#         # How often to save the dense logs
#         "dense_log_frequency": 1,
#     }
# }


# def get_environment():
#     """
#     Returns builded environment with `env_config` config
#     """
#     # return foundation.make_env_instance(**env_config["env_config_dict"])
#     return EnvWrapper(env_config)

if __name__ == "__main__":
    
    tensore = [-10000000, 1e-3, 1,2]
    logit_mask = (torch.softmax(torch.tensor(tensore), dim=0)+1e-3)
    dist = torch.distributions.Categorical(probs=logit_mask)

    # dati = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
    # epochs = 4
    # batch_size = 5
    # # # for i in range(n_agents):
    # # #     print(dati[:i:4])

    # # print(dati[:4])
    # # print(dati[::4])
    # # print(dati[0::4])
    # epochs_selected = list(range(epochs))
    # random.shuffle(epochs_selected)
    # dati: list = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    # for i in epochs_selected:
    #     print(dati[i * batch_size : batch_size + i * batch_size])


#     # env = get_environment()
#     # env.seed(1)
#     # env.env.save_game_object(".")
#     # obs = env.reset()

#     # print(env.observation_space)
#     # print(env.observation_space_pl)
#     # help(env.env)
#     # env.env.previous_episode_replay_log

#     # foundation.utils.save_episode_log(env.env, "./dioawnsdo")
#     # print(env.n_agents)

#     # ablka = {
#     #     '1': 20,
#     #     'p': 'a'
#     # }
#     # print('a' in ablka)

#     # print(env.action_space.__len__)
#     # env.save_game_object(".")

#     i = 0

#     stats = {
#     '0': 0,
#     '1': 0,
#     '2': 0,
#     '3': 0
#     }

#     while i < 1500:
#         start = time.time()
#         obs, rew, _, _ = env.step({
#             '0': np.random.randint(0,49,size=1),
#             '1': np.random.randint(0,49,size=1),
#             '2': np.random.randint(0,49,size=1),
#             '3': np.random.randint(0,49,size=1),
#             'p': [np.random.randint(22,size=1) for _ in range(7)]
#         })

#         if i > 950:
#             end = time.time() - start
#             print(end)

#         i += 1
#     # for key in rew.keys():
#     # if (key != 'p' and rew[key]>0.0):
#     # print(f"key: {key} rew: {rew[key]}")

#     # stats[key]+=1
#     # i+=1

#     # print(f"Results: '0': {stats['0']} , '1': {stats['1']}, '2': {stats['2']},'3': {stats['3']}")
