import gym
# from gym import envs
# print(envs.registry.all())
# env = gym.make("HalfCheetah")
env = gym.make("CarRacing-v1")
env.reset()
# obs, _, _, _ = env.step(gym.spaces.Box([-1],0.2,(3)))
# print(obs[1])
