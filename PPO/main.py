import sys
from ai_economist import foundation
from algorithm import PpoAlgorithm
from env_wrapper import EnvWrapper

env_config = {
    "env_config_dict": {
        # ===== SCENARIO CLASS =====
        # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
        # The environment object will be an instance of the Scenario class.
        "scenario_name": "layout_from_file/simple_wood_and_stone",
        # ===== COMPONENTS =====
        # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
        #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
        #   {component_kwargs} is a dictionary of kwargs passed to the Component class
        # The order in which components reset, step, and generate obs follows their listed order below.
        "components": [
            # (1) Building houses
            (
                "Build",
                {
                    "skill_dist": "pareto",
                    "payment_max_skill_multiplier": 3,
                    "build_labor": 10,
                    "payment": 10,
                },
            ),
            # (2) Trading collectible resources
            (
                "ContinuousDoubleAuction",
                {
                    "max_bid_ask": 10,
                    "order_labor": 0.25,
                    "max_num_orders": 5,
                    "order_duration": 50,
                },
            ),
            # (3) Movement and resource collection
            ("Gather", {"move_labor": 1, "collect_labor": 1, "skill_dist": "pareto"}),
            # (4) Planner
            (
                "PeriodicBracketTax",
                {
                    "period": 100,
                    "bracket_spacing": "us-federal",
                    "usd_scaling": 1000,
                    "disable_taxes": False,
                },
            ),
        ],
        # ===== SCENARIO CLASS ARGUMENTS =====
        # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
        "env_layout_file": "quadrant_25x25_20each_30clump.txt",
        "starting_agent_coin": 10,
        "fixed_four_skill_and_loc": True,
        # ===== STANDARD ARGUMENTS ======
        # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
        "n_agents": 4,  # Number of non-planner agents (must be > 1)
        "world_size": [25, 25],  # [Height, Width] of the env world
        "episode_length": 1000,  # Number of timesteps per episode
        # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
        # Otherwise, the policy selects only 1 action.
        "multi_action_mode_agents": False,
        "multi_action_mode_planner": True,
        # When flattening observations, concatenate scalar & vector observations before output.
        # Otherwise, return observations with minimal processing.
        "flatten_observations": True,
        # When Flattening masks, concatenate each action subspace mask into a single array.
        # Note: flatten_masks = True is required for masking action logits in the code below.
        "flatten_masks": True,
        # How often to save the dense logs
        "dense_log_frequency": 1,
    }
}


def get_environment():
    """
    Returns builded environment with `env_config` config
    """
    # return foundation.make_env_instance(**env_config["env_config_dict"])
    return EnvWrapper(env_config)


if __name__ == "__main__":
    EPOCHS = 1

    env = get_environment()
    obs = env.reset()

    # dummy_env.observation_space_pl,
    # dummy_env.action_space_pl,
    # dummy_env.observation_space,
    # dummy_env.action_space,

    policy_config = {
        'a': {
            'action_space' : 50, # Discete(50)
            'observation_space' : env.observation_space
        },
        # 'p': {
        #     'action_space' : env.action_space_pl,
        #     'observation_space' : env.observation_space_pl
        # }
    }
    
    
    print(obs['0'].keys())
    sys.exit()
    # TODO: make this list automatic
    available_agent_groups = ["a", "p"]

    # Here we can setup a constructor passing a configuration, not done now
    algorithm: PpoAlgorithm = PpoAlgorithm(
        policy_config=policy_config, available_agent_groups=available_agent_groups
    )

    # actions = algorithm.get_actions(obs)
    obs, rew, done, info = env.step(algorithm.get_actions(obs)[0])
    algorithm.train_one_step(env)

    # for i in range(EPOCHS):
    #     # obs, rew, done, info = env.step(PpoAlgorithm.act(obs))
    #     # PpoAlgorithm.train(env)

    #     # print(f"Reward step {EPOCHS}: {rew}")
