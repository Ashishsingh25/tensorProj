import matplotlib.pyplot as plt
import gym
import numpy as np
from tensorflow import keras
import tensorflow as tf
from collections import deque
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.trajectories.trajectory import to_transition
from tf_agents.utils.common import function
from tf_agents.eval.metric_utils import log_metrics
import logging
import PIL
import os

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

# keras.backend.clear_session()
# tf.random.set_seed(42)
# np.random.seed(42)
#
# n_inputs = 4 # == env.observation_space.shape[0]
# model = keras.models.Sequential([
#     keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
#     keras.layers.Dense(1, activation="sigmoid")])

# trying a game with random weights

# def render_policy_net(model, n_max_steps=200, seed=42):
#     env = gym.make("CartPole-v1")
#     env.seed(seed)
#     np.random.seed(seed)
#     obs = env.reset()
#     for step in range(n_max_steps):
#         env.render(mode="rgb_array")
#         left_proba = model.predict(obs.reshape(1, -1))
#         action = int(np.random.rand() > left_proba)
#         obs, reward, done, info = env.step(action)
#         if done:
#             print()
#             print("Reward: ", step)
#             break
#     env.close()
#
# render_policy_net(model)
# Reward:  30

# training with 50 environments

# n_environments = 50
# n_iterations = 5000
#
# envs = [gym.make("CartPole-v1") for _ in range(n_environments)]
# for index, env in enumerate(envs):
#     env.seed(index)
# np.random.seed(42)
# observations = [env.reset() for env in envs]
# optimizer = keras.optimizers.RMSprop()
# loss_fn = keras.losses.binary_crossentropy
#
# for iteration in range(n_iterations):
#     # if angle < 0, we want proba(left) = 1., or else proba(left) = 0.
#     target_probas = np.array([([1.] if obs[2] < 0 else [0.]) for obs in observations])
#     with tf.GradientTape() as tape:
#         left_probas = model(np.array(observations))
#         loss = tf.reduce_mean(loss_fn(target_probas, left_probas))
#     print("\rIteration: {}, Loss: {:.3f}".format(iteration, loss.numpy()), end="")
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     actions = (np.random.rand(n_environments, 1) > left_probas.numpy()).astype(np.int32)
#     for env_index, env in enumerate(envs):
#         obs, reward, done, info = env.step(actions[env_index][0])
#         observations[env_index] = obs if not done else env.reset()
#
# for env in envs:
#     env.close()
# # Iteration: 4999, Loss: 0.094
# render_policy_net(model)
# # Reward:  85

# Policy Gradients

# def play_one_step(env, obs, model, loss_fn):
#     with tf.GradientTape() as tape:
#         left_proba = model(obs[np.newaxis])
#         action = (tf.random.uniform([1, 1]) > left_proba)
#         y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
#         loss = tf.reduce_mean(loss_fn(y_target, left_proba))
#     grads = tape.gradient(loss, model.trainable_variables)
#     obs, reward, done, info = env.step(int(action[0, 0].numpy()))
#     return obs, reward, done, grads
#
# def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
#     all_rewards = []
#     all_grads = []
#     for episode in range(n_episodes):
#         current_rewards = []
#         current_grads = []
#         obs = env.reset()
#         for step in range(n_max_steps):
#             obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
#             current_rewards.append(reward)
#             current_grads.append(grads)
#             if done:
#                 break
#         all_rewards.append(current_rewards)
#         all_grads.append(current_grads)
#     return all_rewards, all_grads
#
# def discount_rewards(rewards, discount_rate):
#     discounted = np.array(rewards)
#     for step in range(len(rewards) - 2, -1, -1):
#         discounted[step] += discounted[step + 1] * discount_rate
#     return discounted
#
# def discount_and_normalize_rewards(all_rewards, discount_rate):
#     all_discounted_rewards = [discount_rewards(rewards, discount_rate)
#                               for rewards in all_rewards]
#     flat_rewards = np.concatenate(all_discounted_rewards)
#     reward_mean = flat_rewards.mean()
#     reward_std = flat_rewards.std()
#     return [(discounted_rewards - reward_mean) / reward_std
#             for discounted_rewards in all_discounted_rewards]
#
# # print(discount_rewards([10, 0, -50], discount_rate=0.8))
# # [-22 -40 -50]
# # print(discount_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_rate=0.8))
# # [array([-0.28435071, -0.86597718, -1.18910299]), array([1.26665318, 1.0727777 ])]
#
# n_iterations = 150
# n_episodes_per_update = 10
# n_max_steps = 200
# discount_rate = 0.95
#
# optimizer = keras.optimizers.Adam(lr=0.01)
# loss_fn = keras.losses.binary_crossentropy
#
# keras.backend.clear_session()
# np.random.seed(42)
# tf.random.set_seed(42)
# model = keras.models.Sequential([
#     keras.layers.Dense(5, activation="elu", input_shape=[4]),
#     keras.layers.Dense(1, activation="sigmoid")])
# env = gym.make("CartPole-v1")
# env.seed(42)
#
# for iteration in range(n_iterations):
#     all_rewards, all_grads = play_multiple_episodes(env, n_episodes_per_update, n_max_steps, model, loss_fn)
#     total_rewards = sum(map(sum, all_rewards))                     # Not shown in the book
#     print("\rIteration: {}, mean rewards: {:.1f}".format(          # Not shown
#         iteration, total_rewards / n_episodes_per_update), end="") # Not shown
#     all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
#     all_mean_grads = []
#     for var_index in range(len(model.trainable_variables)):
#         mean_grads = tf.reduce_mean(
#             [final_reward * all_grads[episode_index][step][var_index]
#              for episode_index, final_rewards in enumerate(all_final_rewards)
#                  for step, final_reward in enumerate(final_rewards)], axis=0)
#         all_mean_grads.append(mean_grads)
#     optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
# env.close()
# # Iteration: 149, mean rewards: 178.3
# model.save("CP_PolicyGrad_model.h5")

# model = keras.models.load_model("CP_PolicyGrad_model.h5")
# render_policy_net(model)
# Reward:  180

### MDP

# transition_probabilities = [ # shape=[s, a, s']
#         [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
#         [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
#         [None, [0.8, 0.1, 0.1], None]]
# rewards = [ # shape=[s, a, s']
#         [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
#         [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
#         [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]]
# possible_actions = [[0, 1, 2], [0, 2], [1]]

# Q_values = np.full((3, 3), -np.inf) # -np.inf for impossible actions
# for state, actions in enumerate(possible_actions):
#     Q_values[state, actions] = 0.0  # for all possible actions

# with gamma 0.9
# gamma = 0.90  # the discount factor
# for iteration in range(50):
#     Q_prev = Q_values.copy()
#     for s in range(3):
#         for a in possible_actions[s]:
#             Q_values[s, a] = np.sum([
#                     transition_probabilities[s][a][sp]
#                     * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
#                 for sp in range(3)])

# print(Q_values)
# [[18.91891892 17.02702702 13.62162162]
#  [ 0.                -inf -4.87971488]
#  [       -inf 50.13365013        -inf]]
# print(np.argmax(Q_values, axis=1))
# [0 0 1]
# with gamma 0.95

# Q_values = np.full((3, 3), -np.inf) # -np.inf for impossible actions
# for state, actions in enumerate(possible_actions):
#     Q_values[state, actions] = 0.0  # for all possible actions
#
# gamma = 0.95  # the discount factor
# for iteration in range(50):
#     Q_prev = Q_values.copy()
#     for s in range(3):
#         for a in possible_actions[s]:
#             Q_values[s, a] = np.sum([
#                     transition_probabilities[s][a][sp]
#                     * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
#                 for sp in range(3)])
# print(Q_values)
# [[21.73304188 20.63807938 16.70138772]
#  [ 0.95462106        -inf  1.01361207]
#  [       -inf 53.70728682        -inf]]
# print(np.argmax(Q_values, axis=1))
# [0 2 1]

### Q-Learning

# def step(state, action):
#     probas = transition_probabilities[state][action]
#     next_state = np.random.choice([0, 1, 2], p=probas)
#     reward = rewards[state][action][next_state]
#     return next_state, reward
#
# def exploration_policy(state):
#     return np.random.choice(possible_actions[state])
#
# np.random.seed(42)
# Q_values = np.full((3, 3), -np.inf)
# for state, actions in enumerate(possible_actions):
#     Q_values[state][actions] = 0
#
# alpha0 = 0.05 # initial learning rate
# decay = 0.005 # learning rate decay
# gamma = 0.90 # discount factor
# state = 0 # initial state
# for iteration in range(10000):
#     action = exploration_policy(state)
#     next_state, reward = step(state, action)
#     next_value = np.max(Q_values[next_state]) # greedy policy at the next step
#     alpha = alpha0 / (1 + iteration * decay)
#     Q_values[state, action] *= 1 - alpha
#     Q_values[state, action] += alpha * (reward + gamma * next_value)
#     state = next_state
# print(Q_values)
# [[18.77621289 17.2238872  13.74543343]
#  [ 0.                -inf -8.00485647]
#  [       -inf 49.40208921        -inf]]
# print(np.argmax(Q_values, axis=1)) # optimal action for each state
# [0 0 1]

### DQN

# keras.backend.clear_session()
# env = gym.make("CartPole-v1")
# env.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)
# input_shape = [4] # == env.observation_space.shape
# n_outputs = 2 # == env.action_space.n
# model = keras.models.Sequential([
#     keras.layers.Dense(32, activation="elu", input_shape=input_shape),
#     keras.layers.Dense(32, activation="elu"),
#     keras.layers.Dense(n_outputs)])

# def epsilon_greedy_policy(state, epsilon=0):
#     if np.random.rand() < epsilon:
#         return np.random.randint(2)
#     else:
#         Q_values = model.predict(state[np.newaxis])
#         return np.argmax(Q_values[0])
#
# # replay_memory = deque(maxlen=2000)
# def sample_experiences(batch_size):
#     indices = np.random.randint(len(replay_memory), size=batch_size)
#     batch = [replay_memory[index] for index in indices]
#     states, actions, rewards, next_states, dones = [
#         np.array([experience[field_index] for experience in batch])
#         for field_index in range(5)]
#     return states, actions, rewards, next_states, dones
#
# def play_one_step(env, state, epsilon):
#     action = epsilon_greedy_policy(state, epsilon)
#     next_state, reward, done, info = env.step(action)
#     replay_memory.append((state, action, reward, next_state, done))
#     return next_state, reward, done, info

# batch_size = 32
# discount_rate = 0.95
# optimizer = keras.optimizers.Adam(lr=1e-3)
# loss_fn = keras.losses.mean_squared_error

# def training_step(batch_size):
#     experiences = sample_experiences(batch_size)
#     states, actions, rewards, next_states, dones = experiences
#     next_Q_values = model.predict(next_states)
#     max_next_Q_values = np.max(next_Q_values, axis=1)
#     target_Q_values = (rewards + (1 - dones) * discount_rate * max_next_Q_values)
#     target_Q_values = target_Q_values.reshape(-1, 1)
#     mask = tf.one_hot(actions, n_outputs)
#     with tf.GradientTape() as tape:
#         all_Q_values = model(states)
#         Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
#         loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
# rewards = []
# best_score = 0
# for episode in range(1000):
#     obs = env.reset()
#     for step in range(200):
#         epsilon = max(1 - episode / 500, 0.01)
#         obs, reward, done, info = play_one_step(env, obs, epsilon)
#         if done:
#             break
#     if episode > 50:
#         training_step(batch_size)
#     rewards.append(step) # Not shown in the book
#     if step > best_score: # Not shown
#         print(step)
#         best_weights = model.get_weights() # Not shown
#         best_score = step # Not shown
#     print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="") # Not shown
# model.set_weights(best_weights)
# model.save("CP_DQN_model.h5")
# plt.plot(rewards)
# plt.show()

# model = keras.models.load_model("CP_DQN_model.h5")
# env.seed(43)
# state = env.reset()
# for step in range(1000):
#     action = epsilon_greedy_policy(state)
#     state, reward, done, info = env.step(action)
#     if done:
#         print()
#         print("Reward: ", step)
#         env.close()
#         break
#     env.render(mode="rgb_array")
# 499

# Double DQN

# keras.backend.clear_session()
# tf.random.set_seed(42)
# np.random.seed(42)
# model = keras.models.Sequential([
#     keras.layers.Dense(32, activation="elu", input_shape=[4]),
#     keras.layers.Dense(32, activation="elu"),
#     keras.layers.Dense(n_outputs)])
# target = keras.models.clone_model(model)
# target.set_weights(model.get_weights())
#
# batch_size = 32
# discount_rate = 0.95
# optimizer = keras.optimizers.Adam(lr=1e-3)
# loss_fn = keras.losses.Huber()
# def training_step(batch_size):
#     experiences = sample_experiences(batch_size)
#     states, actions, rewards, next_states, dones = experiences
#     next_Q_values = model.predict(next_states)
#     best_next_actions = np.argmax(next_Q_values, axis=1)
#     next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
#     next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)
#     target_Q_values = (rewards + (1 - dones) * discount_rate * next_best_Q_values)
#     target_Q_values = target_Q_values.reshape(-1, 1)
#     mask = tf.one_hot(actions, n_outputs)
#     with tf.GradientTape() as tape:
#         all_Q_values = model(states)
#         Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
#         loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
# replay_memory = deque(maxlen=2000)
# rewards = []
# best_score = 0
# for episode in range(1000):
#     obs = env.reset()
#     for step in range(200):
#         epsilon = max(1 - episode / 500, 0.01)
#         obs, reward, done, info = play_one_step(env, obs, epsilon)
#         if done:
#             break
#     if episode > 50:
#         training_step(batch_size)
#     if episode % 50 == 0:
#         target.set_weights(model.get_weights())
#     rewards.append(step)
#     if step > best_score:
#         best_weights = model.get_weights()
#         best_score = step
#     print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="")
#
# model.set_weights(best_weights)
# model.save("CP_DDQN_model.h5")
# plt.plot(rewards)
# plt.show()
#
# env.seed(43)
# state = env.reset()
# for step in range(1000):
#     action = epsilon_greedy_policy(state)
#     state, reward, done, info = env.step(action)
#     if done:
#         print()
#         print("Reward: ", step)
#         env.close()
#         break
#     env.render(mode="rgb_array")
# Reward:  376

### Dueling Double DQN

# keras.backend.clear_session()
# tf.random.set_seed(42)
# np.random.seed(42)
#
# K = keras.backend
# input_states = keras.layers.Input(shape=[4])
# hidden1 = keras.layers.Dense(32, activation="elu")(input_states)
# hidden2 = keras.layers.Dense(32, activation="elu")(hidden1)
# state_values = keras.layers.Dense(1)(hidden2)
# raw_advantages = keras.layers.Dense(n_outputs)(hidden2)
# advantages = raw_advantages - K.max(raw_advantages, axis=1, keepdims=True)
# Q_values = state_values + advantages
# model = keras.models.Model(inputs=[input_states], outputs=[Q_values])
#
# target = keras.models.clone_model(model)
# target.set_weights(model.get_weights())
#
# batch_size = 32
# discount_rate = 0.95
# optimizer = keras.optimizers.Adam(lr=1e-2)
# loss_fn = keras.losses.Huber()
# def training_step(batch_size):
#     experiences = sample_experiences(batch_size)
#     states, actions, rewards, next_states, dones = experiences
#     next_Q_values = model.predict(next_states)
#     best_next_actions = np.argmax(next_Q_values, axis=1)
#     next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
#     next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)
#     target_Q_values = (rewards + (1 - dones) * discount_rate * next_best_Q_values)
#     target_Q_values = target_Q_values.reshape(-1, 1)
#     mask = tf.one_hot(actions, n_outputs)
#     with tf.GradientTape() as tape:
#         all_Q_values = model(states)
#         Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
#         loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
# replay_memory = deque(maxlen=2000)
# rewards = []
# best_score = 0
# for episode in range(1000):
#     obs = env.reset()
#     for step in range(200):
#         epsilon = max(1 - episode / 500, 0.01)
#         obs, reward, done, info = play_one_step(env, obs, epsilon)
#         if done:
#             break
#     if episode > 50:
#         training_step(batch_size)
#     if episode % 200 == 0:
#         target.set_weights(model.get_weights())
#     rewards.append(step)
#     if step > best_score:
#         best_weights = model.get_weights()
#         best_score = step
#     print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon))
#
# model.set_weights(best_weights)
# model.save("CP_DDDQN_model.h5")
# plt.plot(rewards)
# plt.show()
#
# env.seed(43)
# state = env.reset()
# for step in range(1000):
#     action = epsilon_greedy_policy(state)
#     state, reward, done, info = env.step(action)
#     if done:
#         print()
#         print("Reward: ", step)
#         env.close()
#         break
#     env.render(mode="rgb_array")
# Reward:  211

### TF-Agents

# tf.random.set_seed(42)
# np.random.seed(42)

# env = suite_gym.load("Breakout-v4")
# print(env)
# <tf_agents.environments.wrappers.TimeLimit object at 0x000001F61121AFC8>
# print(env.gym)
# <AtariEnv<Breakout-v4>>
# env.seed(42)
# print(env.reset())
# TimeStep(step_type=array(0), reward=array(0., dtype=float32), discount=array(1., dtype=float32), observation=array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#         ..., dtype=uint8))
# plt.imshow(env.render(mode="rgb_array"))
# plt.show()
# print(env.current_time_step())
# print(env.observation_spec())
# BoundedArraySpec(shape=(210, 160, 3), dtype=dtype('uint8'), name='observation', minimum=0, maximum=255)
# print(env.action_spec())
# BoundedArraySpec(shape=(), dtype=dtype('int64'), name='action', minimum=0, maximum=3)
# print(env.time_step_spec())
# TimeStep(step_type=ArraySpec(shape=(), dtype=dtype('int32'), name='step_type'),
#          reward=ArraySpec(shape=(), dtype=dtype('float32'), name='reward'),
#          discount=BoundedArraySpec(shape=(), dtype=dtype('float32'), name='discount', minimum=0.0, maximum=1.0),
#          observation=BoundedArraySpec(shape=(210, 160, 3), dtype=dtype('uint8'),
#                                       name='observation', minimum=0, maximum=255))
# print(env.gym.get_action_meanings())
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
# env.close()

### Environment Wrappers

max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "BreakoutNoFrameskip-v4"

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4])

env.seed(420)
env.reset()
# action = np.array(1, dtype=np.int32)
# time_step = env.step(np.array(1, dtype=np.int32)) # FIRE
# for _ in range(4):
#     time_step = env.step(np.array(3, dtype=np.int32)) # LEFT

def plot_observation(obs):
    # Since there are only 3 color channels, you cannot display 4 frames
    # with one primary color per frame. So this code computes the delta between
    # the current frame and the mean of the other frames, and it adds this delta
    # to the red and blue channels to get a pink color for the current frame.
    obs = obs.astype(np.float32)
    img = obs[..., :3]
    current_frame_delta = np.maximum(obs[..., 3] - obs[..., :3].mean(axis=-1), 0.)
    img[..., 0] += current_frame_delta
    img[..., 2] += current_frame_delta
    img = np.clip(img / 150, 0, 1)
    plt.imshow(img)
    plt.axis("off")

# plt.figure(figsize=(6, 6))
# plot_observation(time_step.observation)
# plt.show()

tf_env = TFPyEnvironment(env)

### Creating DQN

preprocessing_layer = keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]
q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)

# DQN Agent

train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=2.5e-4, decay=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ε
agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000, # <=> 32,000 ALE frames
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()

# Replay buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000)
replay_buffer_observer = replay_buffer.add_batch

# progess message

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")

# training metrics

train_metrics = [tf_metrics.NumberOfEpisodes(),
                 tf_metrics.EnvironmentSteps(),
                 tf_metrics.AverageReturnMetric(),
                 tf_metrics.AverageEpisodeLengthMetric()]

# Collect driver

collect_driver = DynamicStepDriver(tf_env,
                                   agent.collect_policy,
                                   observers=[replay_buffer_observer] + train_metrics,
                                   num_steps=update_period) # collect 4 steps for each training iteration

# Initial experiences, before training

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(2000)],
    num_steps=2000) # <=> 80,000 ALE frames
final_time_step, final_policy_state = init_driver.run()

# 2 sub-episodes, with 3 time steps

# tf.random.set_seed(888) # chosen to show an example of trajectory at the end of an episode
# trajectories, buffer_info = replay_buffer.get_next(sample_batch_size=2, num_steps=3)
# print(trajectories._fields)
# ('step_type', 'observation', 'action', 'policy_info', 'next_step_type', 'reward', 'discount')
# print(trajectories.observation.shape)
# (2, 3, 84, 84, 4)
# time_steps, action_steps, next_time_steps = to_transition(trajectories)
# print(time_steps.observation.shape)
# (2, 2, 84, 84, 4)
# print(trajectories.step_type.numpy())
# [[1 1 1]
#  [1 1 1]]
# plt.figure(figsize=(10, 6.8))
# for row in range(2):
#     for col in range(3):
#         plt.subplot(2, 3, row * 3 + col + 1)
#         plot_observation(trajectories.observation[row, col].numpy())
# plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0.02)
# plt.show()

# dataset

dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

# training

# logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)

train_agent(n_iterations=10000)

frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

prev_lives = tf_env.pyenv.envs[0].ale.lives()
def reset_and_fire_on_life_lost(trajectory):
    global prev_lives
    lives = tf_env.pyenv.envs[0].ale.lives()
    if prev_lives != lives:
        tf_env.reset()
        tf_env.pyenv.envs[0].step(np.array(1, dtype=np.int32))
        prev_lives = lives

watch_driver = DynamicStepDriver(
    tf_env,
    agent.policy,
    observers=[save_frames, reset_and_fire_on_life_lost, ShowProgress(1000)],
    num_steps=1000)
final_time_step, final_policy_state = watch_driver.run()

image_path = os.path.join("D:\\tensor\\Aurelien\\RL","breakout.gif")
frame_images = [PIL.Image.fromarray(frame) for frame in frames[:150]]
frame_images[0].save(image_path, format='GIF',
                     append_images=frame_images[1:],
                     save_all=True,
                     duration=30,
                     loop=0)












