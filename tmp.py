from trainer.environment import get_environment
from trainer.models import LSTMModel
import ray
from copy import deepcopy

if __name__ == "__main__":
    
    env = get_environment("cpu")
    
    obs = env.reset()
    model_planner = LSTMModel(env.observation_space_pl, 22, 'cpu')
    action, logprob = model_planner.act(obs['p'])
    model_planner.evaluate(obs['p'], action)
    # model_agents = LSTMModel(env.observation_space, 50, 'cpu')
