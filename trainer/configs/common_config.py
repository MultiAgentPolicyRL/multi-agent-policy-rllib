"""
Environment Config from Phase1.
"""


def common_params():
    conf = {
        "env": {
            # ===== SCENARIO CLASS =====
            # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
            # The environment object will be an instance of the Scenario class.
            'scenario_name': 'layout_from_file/simple_wood_and_stone',

            # ===== COMPONENTS =====
            # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
            #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
            #   {component_kwargs} is a dictionary of kwargs passed to the Component class
            # The order in which components reset, step, and generate obs follows their listed order below.
            'components': [
                # (1) Building houses
                ('Build', {
                    'skill_dist':                   'pareto',
                    'payment_max_skill_multiplier': 3,
                    'build_labor':                  10,
                    'payment':                      10
                }),
                # (2) Trading collectible resources
                ('ContinuousDoubleAuction', {
                    'max_bid_ask':    10,
                    'order_labor':    0.25,
                    'max_num_orders': 5,
                    'order_duration': 50
                }),
                # (3) Movement and resource collection
                ('Gather', {
                    'move_labor':    1,
                    'collect_labor': 1,
                    'skill_dist':    'pareto'
                }),
                # (4) Planner
                ('PeriodicBracketTax', {
                    'period':          100,
                    'bracket_spacing': 'us-federal',
                    'usd_scaling':     1000,
                    'disable_taxes':   False
                })
            ],

            # ===== SCENARIO CLASS ARGUMENTS =====
            # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
            'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
            'starting_agent_coin': 10,
            'fixed_four_skill_and_loc': True,

            # ===== STANDARD ARGUMENTS ======
            # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
            # Number of non-planner agents (must be > 1)
            'n_agents': 4,
            'world_size': [25, 25],  # [Height, Width] of the env world
            'episode_length': 1000,  # Number of timesteps per episode

            # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
            # Otherwise, the policy selects only 1 action.
            'multi_action_mode_agents': False,
            'multi_action_mode_planner': True,

            # When flattening observations, concatenate scalar & vector observations before output.
            # Otherwise, return observations with minimal processing.
            'flatten_observations': True,
            # When Flattening masks, concatenate each action subspace mask into a single array.
            # Note: flatten_masks = True is required for masking action logits in the code below.
            'flatten_masks': True,

            # How often to save the dense logs
            'dense_log_frequency': 1
        },
        "trainer": {
            "batch_mode": "truncate_episodes",
            "env_config": None,
            "local_tf_session_args": {
                "inter_op_parallelism_threads": 0,
                "intra_op_parallelism_threads": 0,
            },
            "metrics_smoothing_episodes": None,
            "multiagent": None,
            "no_done_at_end": False,
            "num_envs_per_worker": 0,
            "num_gpus": 1,
            "num_gpus_per_worker": 0,
            "num_sgd_iter": 1,
            "num_workers": 0,
            "observation_filter": "NoFilter",
            "rollout_fragment_length": 200,
            "seed": None,
            "sgd_minibatch_size": 1500,
            "shuffle_sequences": True,
            "train_batch_size": 6000,
        },
        "general": {
            "ckpt_frequency_steps": 75000,  # 750000
            "cpus": 12,
            "episodes": 1,  # 25000
            "gpus": 1,
            "restore_tf_weights_agents": '',
            "restore_tf_weights_planner": '',
            "train_planner": False,
        },
    }
    return conf
