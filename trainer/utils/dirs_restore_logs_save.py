from utils import remote, saving
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, format="%(asctime)s %(message)s")
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)


# 🟢
def set_up_dirs_and_maybe_restore(
    run_directory, run_configuration, trainerAgent, trainerPlanner
):
    # === Set up Logging & Saving, or Restore ===
    # All model parameters are always specified in the settings YAML.
    # We do NOT overwrite / reload settings from the previous checkpoint dir.
    # 1. For new runs, the only object that will be loaded from the checkpoint dir
    #    are model weights.
    # 2. For crashed and restarted runs, load_snapshot will reload the full state of
    #    the Trainer(s), including metadata, optimizer, and models.
    (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
    ) = saving.fill_out_run_dir(run_directory)

    # If this is a starting from a crashed run, restore the last trainer snapshot
    if restore_from_crashed_run:
        logger.info(
            "ckpt_dir already exists! Planning to restore using latest snapshot from "
            "earlier (crashed) run with the same ckpt_dir %s",
            ckpt_directory,
        )

        at_loads_agents_ok = saving.load_snapshot(
            trainerAgent, run_directory, load_latest=True, suffix="agent"
        )

        at_loads_planner_ok = saving.load_snapshot(
            trainerPlanner, run_directory, load_latest=True, suffix="planner"
        )

        # at this point, we need at least one good ckpt restored
        if not at_loads_agents_ok and not at_loads_planner_ok:
            logger.fatal(
                "restore_from_crashed_run -> restore_run_dir %s, but no good ckpts "
                "found/loaded!",
                run_directory,
            )
            sys.exit()

        # === Trainer-specific counters ===
        # it's the same for Agents and PPO
        training_step_last_ckpt = (
            int(trainerAgent._timesteps_total) if trainerAgent._timesteps_total else 0
        )
        epis_last_ckpt = (
            int(trainerAgent._episodes_total) if trainerAgent._episodes_total else 0
        )

    else:

        logger.info("Not restoring trainer...")
        # === Trainer-specific counters ===
        training_step_last_ckpt = 0
        epis_last_ckpt = 0

        # == For new runs, load only tf checkpoint weights ==

        # Agents
        starting_weights_path_agents = run_configuration["general"].get(
            "restore_tf_weights_agents", ""
        )
        if starting_weights_path_agents:
            logger.info("Restoring agents TF weights...")
            saving.load_tf_model_weights(trainerAgent, starting_weights_path_agents)
        else:
            logger.info("Starting with fresh agent TF weights.")

        # Planner
        starting_weights_path_planner = run_configuration["general"].get(
            "restore_tf_weights_planner", ""
        )
        if starting_weights_path_planner:
            logger.info("Restoring planner TF weights...")
            saving.load_tf_model_weights(trainerPlanner, starting_weights_path_planner)
        else:
            logger.info("Starting with fresh planner TF weights.")

    return (
        dense_log_directory,
        ckpt_directory,
        restore_from_crashed_run,
        training_step_last_ckpt,
        epis_last_ckpt,
    )


# 🟢 changed agent_trainer, added planner_trainer and ifPlanner
def maybe_store_dense_log(
    agent_trainer,
    planner_trainer,
    result_dict,
    dense_log_freq,
    dense_log_directory,
    ifPlanner,
):
    if result_dict["episodes_this_iter"] > 0 and dense_log_freq > 0:
        episodes_per_replica = (
            result_dict["episodes_total"] // result_dict["episodes_this_iter"]
        )
        if episodes_per_replica == 1 or (episodes_per_replica % dense_log_freq) == 0:
            log_dir = os.path.join(
                dense_log_directory,
                "logs_{:016d}".format(result_dict["timesteps_total"]),
            )
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)

            saving.write_dense_logs(agent_trainer, log_dir)

            if ifPlanner:
                saving.write_dense_logs(planner_trainer, log_dir, "planner")

            logger.info(">> Wrote dense logs to: %s", log_dir)


# 🟢 changed agent_trainer, added planner_trainer and ifPlanner
def maybe_save(
    agent_trainer,
    planner_trainer,
    result_dict,
    ckpt_freq,
    ckpt_directory,
    trainer_step_last_ckpt,
    ifPlanner,
):
    global_step = result_dict["timesteps_total"]

    # Check if saving this iteration
    if (
        result_dict["episodes_this_iter"] > 0
    ):  # Don't save if midway through an episode.

        if ckpt_freq > 0:
            if global_step - trainer_step_last_ckpt >= ckpt_freq:
                saving.save_snapshot(agent_trainer, ckpt_directory, suffix="agent")
                saving.save_tf_model_weights(
                    agent_trainer, ckpt_directory, global_step, suffix="agent"
                )

                if ifPlanner:
                    saving.save_snapshot(
                        planner_trainer, ckpt_directory, suffix="planner"
                    )
                    saving.save_tf_model_weights(
                        planner_trainer, ckpt_directory, global_step, suffix="planner"
                    )

                trainer_step_last_ckpt = int(global_step)

                logger.info("Checkpoint saved @ step %d", global_step)

    return trainer_step_last_ckpt
