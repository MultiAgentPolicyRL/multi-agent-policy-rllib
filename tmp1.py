from src.common.env import get_environment

env = get_environment()
obs = env.reset()['0']
print(len(obs))