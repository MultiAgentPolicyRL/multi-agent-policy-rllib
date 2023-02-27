from trainer.environment import get_environment
import ray
from copy import deepcopy

if __name__ == "__main__":
    ray.init()
    env = get_environment("cpu")
    # obs = env.reset()

    # for key in obs.keys():
    #     for object in obs[key].keys():
    #         print(f"{key} {object} {obs[key][object].shape}")

    # Define the Counter actor.
    @ray.remote
    class Worker:
        def __init__(self, env):
            self.env = deepcopy(env)
            self.obs = None

        def reset(self):
            self.obs = self.env.reset()

        def get_obs(self,):
            return self.obs

    # Create a Counter actor.
    c = Worker.remote(env)

    # Submit calls to the actor. These calls run asynchronously but in
    # submission order on the remote actor process.
    c.reset.remote()

    # Retrieve final actor state.
    print(ray.get(c.get_obs.remote()))
    # -> 10