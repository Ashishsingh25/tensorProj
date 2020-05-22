import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LayerNormalization

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)

def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])

# np.random.seed(42)
# n_steps = 50
# series = generate_time_series(10000, n_steps + 1)
# X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
# X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
# X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
# print(X_train.shape, y_train.shape)
# (7000, 50, 1) (7000, 1)

# just predict the last observed value

# y_pred = X_valid[:, -1]
# print(np.mean(keras.losses.mean_squared_error(y_valid, y_pred)))
# 0.020211367
# plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
# plt.show()

# Linear predictions

# np.random.seed(42)
# tf.random.set_seed(42)
#
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[50, 1]),
#     keras.layers.Dense(1)])
# print(model.summary())

# model.compile(loss="mse", optimizer="adam")
# history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
# print(model.get_weights()[0])
# plt.plot(model.get_weights()[0])
# plt.plot(model.get_weights()[0],'bo')
# print(model.evaluate(X_valid, y_valid))
# # 0.004145486194640398
# y_pred = model.predict(X_valid)
# plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
# plt.show()

# Simple RNN

# model = keras.models.Sequential([
#     keras.layers.SimpleRNN(1, input_shape=[None, 1])])
# print(model.summary())
#
# optimizer = keras.optimizers.Adam(lr=0.005)
# model.compile(loss="mse", optimizer=optimizer)
# history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
# print(model.evaluate(X_valid, y_valid))
# # 0.010885455287992955
# y_pred = model.predict(X_valid)
# plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
# plt.show()

# Deep RNN

# model = keras.models.Sequential([
#     keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
#     keras.layers.SimpleRNN(20),
#     keras.layers.Dense(1)])
# print(model.summary())

# model.compile(loss="mse", optimizer="adam")
# history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
# print(model.evaluate(X_valid, y_valid))
# # 0.002541951509192586
# y_pred = model.predict(X_valid)
# plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
# plt.show()

### Forecasting 10 Steps

def plot_multiple_forecasts(X, Y, Y_pred):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast", markersize=10)
    plt.axis([0, n_steps + ahead, -1, 1])
    plt.legend(fontsize=14)

# using prediction as input for next prediction

# np.random.seed(43)
# series = generate_time_series(1, n_steps + 10)
# X_new, Y_new = series[:, :n_steps], series[:, n_steps:]
# X = X_new
# for step_ahead in range(10):
#     y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
#     X = np.concatenate([X, y_pred_one], axis=1)
# Y_pred = X[:, n_steps:]
# print(tf.concat((Y_pred, Y_new), axis=2))
# tf.Tensor(
# [[[ 0.60153234  0.64557177]
#   [ 0.63857627  0.6562027 ]
#   [ 0.5946282   0.65506256]
#   [ 0.48533857  0.5576619 ]
#   [ 0.31978258  0.39075595]
#   [ 0.13677266  0.19883814]
#   [-0.01865985 -0.0130802 ]
#   [-0.12659991 -0.15594868]
#   [-0.18722412 -0.18422735]
#   [-0.20886402 -0.2669426 ]]], shape=(1, 10, 2), dtype=float32)
# plot_multiple_forecasts(X_new, Y_new, Y_pred)
# plt.show()

# predict the next 10 values

# np.random.seed(42)
# n_steps = 50
# series = generate_time_series(10000, n_steps + 10)
# X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
# X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
# X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]

# linear model

# np.random.seed(42)
# tf.random.set_seed(42)
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[50, 1]),
#     keras.layers.Dense(10)])
# model.compile(loss="mse", optimizer="adam")
# history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
# loss: 0.0189 - val_loss: 0.0188

# Deep RNN model

# model = keras.models.Sequential([
#     keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
#     keras.layers.SimpleRNN(20),
#     keras.layers.Dense(10)
# ])
# model.compile(loss="mse", optimizer="adam")
# history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
# # loss: 0.0093 - val_loss: 0.0093
# np.random.seed(43)
# series = generate_time_series(1, 50 + 10)
# X_new, Y_new = series[:, :50, :], series[:, -10:, :]
# Y_pred = model.predict(X_new)[..., np.newaxis]
# print(tf.concat((Y_pred, Y_new), axis=2))
# # tf.Tensor(
# # [[[ 0.5681278   0.64557177]
# #   [ 0.5715649   0.6562027 ]
# #   [ 0.49534318  0.65506256]
# #   [ 0.42914495  0.5576619 ]
# #   [ 0.338508    0.39075595]
# #   [ 0.2646662   0.19883814]
# #   [ 0.19067888 -0.0130802 ]
# #   [ 0.12743853 -0.15594868]
# #   [ 0.0988944  -0.18422735]
# #   [ 0.00398464 -0.2669426 ]]], shape=(1, 10, 2), dtype=float32)
# plot_multiple_forecasts(X_new, Y_new, Y_pred)
# plt.show()

# RNN that predicts the next 10 steps at each time step

np.random.seed(42)
n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train = series[:7000, :n_steps]
X_valid = series[7000:9000, :n_steps]
X_test = series[9000:, :n_steps]
Y = np.empty((10000, n_steps, 10))
for step_ahead in range(1, 10 + 1):
    Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

# np.random.seed(42)
# tf.random.set_seed(42)
# model = keras.models.Sequential([
#     keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
#     keras.layers.SimpleRNN(20, return_sequences=True),
#     keras.layers.TimeDistributed(keras.layers.Dense(10))])

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

# model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01), metrics=[last_time_step_mse])
# history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
# loss: 0.0182 - last_time_step_mse: 0.0065 - val_loss: 0.0188 - val_last_time_step_mse: 0.0077

# np.random.seed(43)
# series = generate_time_series(1, 50 + 10)
# X_new, Y_new = series[:, :50, :], series[:, 50:, :]
# Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]
# print(tf.concat((Y_pred, Y_new), axis=2))
# # tf.Tensor(
# # [[[ 0.5302257   0.64557177]
# #   [ 0.61683655  0.6562027 ]
# #   [ 0.6010575   0.65506256]
# #   [ 0.5211494   0.5576619 ]
# #   [ 0.38858134  0.39075595]
# #   [ 0.23027211  0.19883814]
# #   [ 0.08102068 -0.0130802 ]
# #   [-0.03815448 -0.15594868]
# #   [-0.1189469  -0.18422735]
# #   [-0.17355414 -0.2669426 ]]], shape=(1, 10, 2), dtype=float32)
# plot_multiple_forecasts(X_new, Y_new, Y_pred)
# plt.show()

### Batch Norm

# np.random.seed(42)
# tf.random.set_seed(42)
# model = keras.models.Sequential([
#     keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
#     keras.layers.BatchNormalization(),
#     keras.layers.SimpleRNN(20, return_sequences=True),
#     keras.layers.BatchNormalization(),
#     keras.layers.TimeDistributed(keras.layers.Dense(10))])
#
# model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
# history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
# loss: 0.0282 - last_time_step_mse: 0.0162 - val_loss: 0.0323 - val_last_time_step_mse: 0.0214

### Layer Norm

# class LNSimpleRNNCell(keras.layers.Layer):
#     def __init__(self, units, activation="tanh", **kwargs):
#         super().__init__(**kwargs)
#         self.state_size = units
#         self.output_size = units
#         self.simple_rnn_cell = keras.layers.SimpleRNNCell(units,
#                                                           activation=None)
#         self.layer_norm = LayerNormalization()
#         self.activation = keras.activations.get(activation)
#     def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
#         if inputs is not None:
#             batch_size = tf.shape(inputs)[0]
#             dtype = inputs.dtype
#         return [tf.zeros([batch_size, self.state_size], dtype=dtype)]
#     def call(self, inputs, states):
#         outputs, new_states = self.simple_rnn_cell(inputs, states)
#         norm_outputs = self.activation(self.layer_norm(outputs))
#         return norm_outputs, [norm_outputs]
#
np.random.seed(42)
tf.random.set_seed(42)
# model = keras.models.Sequential([
#     keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True,
#                      input_shape=[None, 1]),
#     keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True),
#     keras.layers.TimeDistributed(keras.layers.Dense(10))])
#
# model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
# history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
# loss: 0.0273 - last_time_step_mse: 0.0149 - val_loss: 0.0268 - val_last_time_step_mse: 0.0141

### LSTMs

# model = keras.models.Sequential([
#     keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
#     keras.layers.LSTM(20, return_sequences=True),
#     keras.layers.TimeDistributed(keras.layers.Dense(10))])
#
# model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
# history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
# # loss: 0.0239 - last_time_step_mse: 0.0089 - val_loss: 0.0240 - val_last_time_step_mse: 0.0086
# print(model.evaluate(X_valid, Y_valid))
# # [0.02401665359735489, 0.008551412]
# np.random.seed(43)
# series = generate_time_series(1, 50 + 10)
# X_new, Y_new = series[:, :50, :], series[:, 50:, :]
# Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]
# # print(tf.concat((Y_pred, Y_new), axis=2))
# # tf.Tensor(
# # [[[ 0.54658085  0.64557177]
# #   [ 0.6022002   0.6562027 ]
# #   [ 0.5455934   0.65506256]
# #   [ 0.4606139   0.5576619 ]
# #   [ 0.33550242  0.39075595]
# #   [ 0.20063888  0.19883814]
# #   [ 0.08716021 -0.0130802 ]
# #   [-0.01352238 -0.15594868]
# #   [-0.10610919 -0.18422735]
# #   [-0.18007736 -0.2669426 ]]], shape=(1, 10, 2), dtype=float32)
# plot_multiple_forecasts(X_new, Y_new, Y_pred)
# plt.show()

### GRU

# model = keras.models.Sequential([
#     keras.layers.GRU(20, return_sequences=True, input_shape=[None, 1]),
#     keras.layers.GRU(20, return_sequences=True),
#     keras.layers.TimeDistributed(keras.layers.Dense(10))])
#
# model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
# history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
# # loss: 0.0242 - last_time_step_mse: 0.0106 - val_loss: 0.0241 - val_last_time_step_mse: 0.0103
# print(model.evaluate(X_valid, Y_valid))
# # [0.024071006283164026, 0.010298316]
# np.random.seed(43)
# series = generate_time_series(1, 50 + 10)
# X_new, Y_new = series[:, :50, :], series[:, 50:, :]
# Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]
# print(tf.concat((Y_pred, Y_new), axis=2))
# # tf.Tensor(
# # [[[ 0.59997356  0.64557177]
# #   [ 0.5988938   0.6562027 ]
# #   [ 0.5366406   0.65506256]
# #   [ 0.46059778  0.5576619 ]
# #   [ 0.35147652  0.39075595]
# #   [ 0.24302815  0.19883814]
# #   [ 0.1510677  -0.0130802 ]
# #   [ 0.04900258 -0.15594868]
# #   [-0.06261393 -0.18422735]
# #   [-0.1516203  -0.2669426 ]]], shape=(1, 10, 2), dtype=float32)
# plot_multiple_forecasts(X_new, Y_new, Y_pred)
# plt.show()

### 1D Convolutional Layers

# model = keras.models.Sequential([
#     keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid", input_shape=[None, 1]),
#     keras.layers.GRU(20, return_sequences=True),
#     keras.layers.GRU(20, return_sequences=True),
#     keras.layers.TimeDistributed(keras.layers.Dense(10))])
#
# model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
# history = model.fit(X_train, Y_train[:, 3::2], epochs=20, validation_data=(X_valid, Y_valid[:, 3::2]))
# loss: 0.0175 - last_time_step_mse: 0.0072 - val_loss: 0.0173 - val_last_time_step_mse: 0.0067

### WaveNet

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[None, 1]))
for rate in (1, 2, 4, 8) * 2:
    model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate))
model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
# loss: 0.0198 - last_time_step_mse: 0.0081 - val_loss: 0.0197 - val_last_time_step_mse: 0.0080
