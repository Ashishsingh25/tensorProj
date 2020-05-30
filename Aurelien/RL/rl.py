import matplotlib.pyplot as plt
import gym
import numpy as np
from tensorflow import keras
import tensorflow as tf

# env = gym.make('CartPole-v1')
# env.seed(42)
# obs = env.reset()
# print(obs)
# [-0.01258566 -0.00156614  0.04207708 -0.00180545]
# env.render()
# img = env.render(mode="rgb_array")
# print(img.shape)

# def plot_environment(env, figsize=(5,4)):
#     plt.figure(figsize=figsize)
#     img = env.render(mode="rgb_array")
#     plt.imshow(img)
#     plt.axis("off")
#     return img
# plot_environment(env)
# plt.show()

# print(env.action_space)
# # Discrete(2)
# action = 1  # accelerate right
# obs, reward, done, info = env.step(action)
# print(obs, reward, done, info)
# # [-0.01261699  0.19292789  0.04204097 -0.28092127] 1.0 False {}

### Simple policy

# env.seed(42)
# def basic_policy(obs):
#     angle = obs[2]
#     return 0 if angle < 0 else 1
#
# totals = []
# for episode in range(500):
#     episode_rewards = 0
#     obs = env.reset()
#     for step in range(200):
#         action = basic_policy(obs)
#         obs, reward, done, info = env.step(action)
#         episode_rewards += reward
#         if done:
#             break
#     totals.append(episode_rewards)
#
# # print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
# # 41.718 8.858356280936096 24.0 68.0
#
# # visualize one episode
#
# env.seed(42)
# obs = env.reset()
# for step in range(200):
#     env.render(mode="rgb_array")
#     action = basic_policy(obs)
#     obs, reward, done, info = env.step(action)
#     if done:
#         print(step)
#         break
# env.close()

### NN policy

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

n_inputs = 4 # == env.observation_space.shape[0]
model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid")])

# trying a game with random weights

def render_policy_net(model, n_max_steps=200, seed=42):
    env = gym.make("CartPole-v1")
    env.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    for step in range(n_max_steps):
        env.render(mode="rgb_array")
        left_proba = model.predict(obs.reshape(1, -1))
        action = int(np.random.rand() > left_proba)
        obs, reward, done, info = env.step(action)
        if done:
            print()
            print("Reward: ", step)
            break
    env.close()
#
render_policy_net(model)
# Reward:  30

# training with 50 environments

n_environments = 50
n_iterations = 5000

envs = [gym.make("CartPole-v1") for _ in range(n_environments)]
for index, env in enumerate(envs):
    env.seed(index)
np.random.seed(42)
observations = [env.reset() for env in envs]
optimizer = keras.optimizers.RMSprop()
loss_fn = keras.losses.binary_crossentropy

for iteration in range(n_iterations):
    # if angle < 0, we want proba(left) = 1., or else proba(left) = 0.
    target_probas = np.array([([1.] if obs[2] < 0 else [0.]) for obs in observations])
    with tf.GradientTape() as tape:
        left_probas = model(np.array(observations))
        loss = tf.reduce_mean(loss_fn(target_probas, left_probas))
    print("\rIteration: {}, Loss: {:.3f}".format(iteration, loss.numpy()), end="")
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    actions = (np.random.rand(n_environments, 1) > left_probas.numpy()).astype(np.int32)
    for env_index, env in enumerate(envs):
        obs, reward, done, info = env.step(actions[env_index][0])
        observations[env_index] = obs if not done else env.reset()

for env in envs:
    env.close()
# Iteration: 4999, Loss: 0.094
render_policy_net(model)
# Reward:  85