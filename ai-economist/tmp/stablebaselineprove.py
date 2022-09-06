from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
