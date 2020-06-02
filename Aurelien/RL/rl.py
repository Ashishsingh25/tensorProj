import matplotlib.pyplot as plt
import gym
import numpy as np
from tensorflow import keras
import tensorflow as tf
from collections import deque

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

transition_probabilities = [ # shape=[s, a, s']
        [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
        [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
        [None, [0.8, 0.1, 0.1], None]]
rewards = [ # shape=[s, a, s']
        [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
        [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]]
possible_actions = [[0, 1, 2], [0, 2], [1]]

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

keras.backend.clear_session()
env = gym.make("CartPole-v1")
env.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n
model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)])

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

replay_memory = deque(maxlen=2000)
def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

batch_size = 32
discount_rate = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error

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

model = keras.models.load_model("CP_DQN_model.h5")
env.seed(43)
state = env.reset()
for step in range(1000):
    action = epsilon_greedy_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        print()
        print("Reward: ", step)
        env.close()
        break
    env.render(mode="rgb_array")

















