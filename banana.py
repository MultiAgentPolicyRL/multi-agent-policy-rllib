from trainer.environment import get_environment

if __name__ == '__main__':
    env = get_environment('cpu')
    obs = env.reset()
    for i in range(1001):
        _, _, done, _ = env.step({})
        print(done)