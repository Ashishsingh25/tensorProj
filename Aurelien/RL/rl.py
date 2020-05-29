import gym

env = gym.make('CartPole-v1')
env.seed(42)
obs = env.reset()
print(obs)
print(env.render())


