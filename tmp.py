from trainer.environment import get_environment

if __name__ == "__main__":
    env = get_environment("cpu")
    obs = env.reset()

    for key in obs.keys():
        for object in obs[key].keys():
            print(f"{key} {object} {obs[key][object].shape}")
