from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,
                                                              housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

### Custom loss function

# def huber_fn(y_true, y_pred):
#     error = y_true - y_pred
#     is_small_error = tf.abs(error) < 1
#     squared_loss = tf.square(error) / 2
#     linear_loss  = tf.abs(error) - 0.5
#     return tf.where(is_small_error, squared_loss, linear_loss)

input_shape = X_train.shape[1:]
# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal", input_shape=input_shape),
#     keras.layers.Dense(1),])
# model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])
# model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid))
# loss: 0.2159 - mae: 0.5153 - val_loss: 0.1971 - val_mae: 0.4887

# Saving/Loading Models with Custom Objects
# model.save("my_model_with_a_custom_loss.h5")
# model = keras.models.load_model("my_model_with_a_custom_loss.h5", custom_objects={"huber_fn": huber_fn})
# model.fit(X_train_scaled, y_train, epochs=2, validation_data=(X_valid_scaled, y_valid))
# loss: 0.2084 - mae: 0.5004 - val_loss: 0.2009 - val_mae: 0.4809

### Custom Activation Functions, Initializers, Regularizers, and Constraints

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# def my_softplus(z): # return value is just tf.nn.softplus(z)
#     return tf.math.log(tf.exp(z) + 1.0)
#
# def my_glorot_initializer(shape, dtype=tf.float32):
#     stddev = tf.sqrt(2. / (shape[0] + shape[1]))
#     return tf.random.normal(shape, stddev=stddev, dtype=dtype)
#
# def my_l1_regularizer(weights):
#     return tf.reduce_sum(tf.abs(0.01 * weights))
#
# def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
#     return tf.where(weights < 0., tf.zeros_like(weights), weights)

# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal", input_shape=input_shape),
#     keras.layers.Dense(1, activation=my_softplus,
#                        kernel_regularizer=my_l1_regularizer,
#                        kernel_constraint=my_positive_weights,
#                        kernel_initializer=my_glorot_initializer)])
# model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
# model.fit(X_train_scaled, y_train, epochs=10, validation_data=(X_valid_scaled, y_valid))
# loss: 0.4758 - mae: 0.4781 - val_loss: 0.8552 - val_mae: 0.4745
# model.save("my_model_with_many_custom_parts.h5")
# model = keras.models.load_model("my_model_with_many_custom_parts.h5",
#     custom_objects={"my_l1_regularizer": my_l1_regularizer,
#        "my_positive_weights": my_positive_weights,
#        "my_glorot_initializer": my_glorot_initializer,
#        "my_softplus": my_softplus,})
# model.fit(X_train_scaled, y_train, epochs=10, validation_data=(X_valid_scaled, y_valid))
# loss: 0.4637 - mae: 0.4728 - val_loss: 0.6212 - val_mae: 0.4643

### Custom Metrics

# def create_huber(threshold=1.0):
#     def huber_fn(y_true, y_pred):
#         error = y_true - y_pred
#         is_small_error = tf.abs(error) < threshold
#         squared_loss = tf.square(error) / 2
#         linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
#         return tf.where(is_small_error, squared_loss, linear_loss)
#     return huber_fn
#
# model = keras.models.Sequential([keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
#                                                     input_shape=input_shape),
#                                  keras.layers.Dense(1)])
# model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])
# model.fit(X_train_scaled, y_train, epochs=2)
# loss: 0.5862 - huber_fn: 0.2682

# model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[create_huber(2.0)])
# sample_weight = np.random.rand(len(y_train))
# history = model.fit(X_train_scaled, y_train, epochs=2, sample_weight=sample_weight)
# loss: 0.1286 - huber_fn: 0.2604
# print(history.history["loss"][0], history.history["huber_fn"][0],history.history["huber_fn"][0] * sample_weight.mean())
# 0.4329958712356446 0.87545186 0.4344403442729984

# Streaming metrics

# precision = keras.metrics.Precision()
# precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
# print(precision.result())
# # tf.Tensor(0.8, shape=(), dtype=float32)
# precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])
# print(precision.result())
# # tf.Tensor(0.5, shape=(), dtype=float32)

# class HuberMetric(keras.metrics.Metric):
#     def __init__(self, threshold=1.0, **kwargs):
#         super().__init__(**kwargs) # handles base args (e.g., dtype)
#         self.threshold = threshold
#         self.huber_fn = create_huber(threshold)
#         self.total = self.add_weight("total", initializer="zeros")
#         self.count = self.add_weight("count", initializer="zeros")
#     def huber_fn(self, y_true, y_pred): # workaround
#         error = y_true - y_pred
#         is_small_error = tf.abs(error) < self.threshold
#         squared_loss = tf.square(error) / 2
#         linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
#         return tf.where(is_small_error, squared_loss, linear_loss)
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         metric = self.huber_fn(y_true, y_pred)
#         self.total.assign_add(tf.reduce_sum(metric))
#         self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
#     def result(self):
#         return self.total / self.count
#     def get_config(self):
#         base_config = super().get_config()
#         return {**base_config, "threshold": self.threshold}

# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal", input_shape=input_shape),
#     keras.layers.Dense(1)])
# model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[HuberMetric(2.0)])
# model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)
# # loss: 0.2564 - huber_metric: 0.2564

### Custom Layers

# class MyDense(keras.layers.Layer):
#     def __init__(self, units, activation=None, **kwargs):
#         super().__init__(**kwargs)
#         self.units = units
#         self.activation = keras.activations.get(activation)
#
#     def build(self, batch_input_shape):
#         self.kernel = self.add_weight(name="kernel", shape=[batch_input_shape[-1], self.units],
#                                       initializer="glorot_normal")
#         self.bias = self.add_weight(name="bias", shape=[self.units], initializer="zeros")
#         super().build(batch_input_shape) # must be at the end
#
#     def call(self, X):
#         return self.activation(X @ self.kernel + self.bias)
#
#     def compute_output_shape(self, batch_input_shape):
#         return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])
#
#     def get_config(self):
#         base_config = super().get_config()
#         return {**base_config, "units": self.units,
#                 "activation": keras.activations.serialize(self.activation)}
#
# model = keras.models.Sequential([MyDense(30, activation="relu", input_shape=input_shape),
#                                  MyDense(1)])
# model.compile(loss="mse", optimizer="nadam")
# model.fit(X_train_scaled, y_train, epochs=10, validation_data=(X_valid_scaled, y_valid))
# # loss: 0.3734 - val_loss: 0.4260
# print(model.evaluate(X_test_scaled, y_test))
# # loss: 0.3649

### Custom Models

# class ResidualBlock(keras.layers.Layer):
#     def __init__(self, n_layers, n_neurons, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden = [keras.layers.Dense(n_neurons, activation="elu", kernel_initializer="he_normal")
#                        for _ in range(n_layers)]
#
#     def call(self, inputs):
#         Z = inputs
#         for layer in self.hidden:
#             Z = layer(Z)
#         return inputs + Z
#
# class ResidualRegressor(keras.models.Model):
#     def __init__(self, output_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden1 = keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal")
#         self.block1 = ResidualBlock(2, 30)
#         self.block2 = ResidualBlock(2, 30)
#         self.out = keras.layers.Dense(output_dim)
#
#     def call(self, inputs):
#         Z = self.hidden1(inputs)
#         for _ in range(1 + 3):
#             Z = self.block1(Z)
#         Z = self.block2(Z)
#         return self.out(Z)
#
#
# model = ResidualRegressor(1)
# model.compile(loss="mse", optimizer="nadam")
# history = model.fit(X_train_scaled, y_train, epochs=10)
# # loss: 0.5358
# score = model.evaluate(X_test_scaled, y_test)
# print(score)
# # 0.430588229866915
# X_new_scaled = X_test_scaled[:5]
# y_pred = model.predict(X_new_scaled)
# print(y_test[:5])
# # [[0.477  ][0.458  ][5.00001][2.186  ][2.78   ]]
# print(y_pred)
# # [[0.18434484][1.4001691 ][4.782007  ][2.374351  ][3.2077434 ]]

### Losses and Metrics Based on Model Internals

# class ReconstructingRegressor(keras.models.Model):
#     def __init__(self, output_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.hidden = [keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal") for _ in range(5)]
#         self.out = keras.layers.Dense(output_dim)
#
#     def build(self, batch_input_shape):
#         n_inputs = batch_input_shape[-1]
#         self.reconstruct = keras.layers.Dense(n_inputs)
#         super().build(batch_input_shape)
#
#     def call(self, inputs, training=None):
#         Z = inputs
#         for layer in self.hidden:
#             Z = layer(Z)
#         reconstruction = self.reconstruct(Z)
#         recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
#         self.add_loss(0.05 * recon_loss)
#         return self.out(Z)
#
# model = ReconstructingRegressor(1)
# model.compile(loss="mse", optimizer="nadam")
# history = model.fit(X_train_scaled, y_train, epochs=10)
# # loss: 0.3222
# print(model.evaluate(X_test_scaled, y_test))
# # 0.33062958742759024
# y_pred = model.predict(X_test_scaled)
# print(y_test[:5])
# # [[0.477  ][0.458  ][5.00001][2.186  ][2.78   ]]
# print(y_pred[:5])
# # [[0.45005453][0.9927232 ][4.018667  ][2.3474922 ][2.7517285 ]]

### Gradients with Autodiff

# def f(w1, w2):
#     return 3 * w1 ** 2 + 2 * w1 * w2
# w1, w2 = 5, 3
# eps = 1e-6
# print((f(w1 + eps, w2) - f(w1, w2)) / eps)
# # 36.000003007075065
# print((f(w1, w2 + eps) - f(w1, w2)) / eps)
# # 10.000000003174137

# w1, w2 = tf.Variable(5.), tf.Variable(3.)
# with tf.GradientTape() as tape:
#     z = f(w1, w2)
# gradients = tape.gradient(z, [w1, w2])
# print(gradients)
# [<tf.Tensor: shape=(), dtype=float32, numpy=36.0>, <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]

### Custom Training Loops

# l2_reg = keras.regularizers.l2(0.05)
# model = keras.models.Sequential([keras.layers.Dense(30, activation="elu", kernel_initializer="he_normal",
#                                                     kernel_regularizer=l2_reg),
#     keras.layers.Dense(1, kernel_regularizer=l2_reg)])
#
# def random_batch(X, y, batch_size=32):
#     idx = np.random.randint(len(X), size=batch_size)
#     return X[idx], y[idx]
#
# def print_status_bar(iteration, total, loss, metrics=None):
#     metrics = " - ".join(["{}: {:.4f}".format(m.name, m.result()) for m in [loss] + (metrics or [])])
#     end = "" if iteration < total else "\n"
#     print("\r{}/{} - ".format(iteration, total) + metrics, end=end)
#
# n_epochs = 5
# batch_size = 32
# n_steps = len(X_train) // batch_size
# optimizer = keras.optimizers.Nadam(lr=0.01)
# loss_fn = keras.losses.mean_squared_error
# mean_loss = keras.metrics.Mean()
# metrics = [keras.metrics.MeanAbsoluteError()]
#
# for epoch in range(1, n_epochs + 1):
#     print("Epoch {}/{}".format(epoch, n_epochs))
#     for step in range(1, n_steps + 1):
#         X_batch, y_batch = random_batch(X_train_scaled, y_train)
#         with tf.GradientTape() as tape:
#             y_pred = model(X_batch)
#             main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
#             loss = tf.add_n([main_loss] + model.losses)
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         for variable in model.variables:
#             if variable.constraint is not None:
#                 variable.assign(variable.constraint(variable))
#         mean_loss(loss)
#         for metric in metrics:
#             metric(y_batch, y_pred)
#         print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)
#     print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
#     for metric in [mean_loss] + metrics:
#         metric.reset_states()

# Epoch 1/5
# WARNING:tensorflow:Layer dense is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
#
# If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
#
# To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
#
# 2020-05-15 20:08:10.871160: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_10.dll
# 11610/11610 - mean: 1.3955 - mean_absolute_error: 0.5722
# Epoch 2/5
# 11610/11610 - mean: 0.6774 - mean_absolute_error: 0.5280
# Epoch 3/5
# 11610/11610 - mean: 0.6351 - mean_absolute_error: 0.5177
# Epoch 4/5
# 11610/11610 - mean: 0.6384 - mean_absolute_error: 0.5181
# Epoch 5/5
# 11610/11610 - mean: 0.6440 - mean_absolute_error: 0.5222
#
# Process finished with exit code 0

### TensorFlow Functions

# def cube(x):
#     return x ** 3
# print(cube(2))
# # 8
# print(cube(tf.constant(2.0)))
# # tf.Tensor(8.0, shape=(), dtype=float32)
#
# tf_cube = tf.function(cube)
# print(tf_cube)
# # <tensorflow.python.eager.def_function.Function object at 0x0000020EFD8E2D08>
# print(tf_cube(2))
# # tf.Tensor(8, shape=(), dtype=int32)
# print(tf_cube(tf.constant(2.0)))
# # tf.Tensor(8.0, shape=(), dtype=float32)
#
# @tf.function
# def add_10(x):
#     for i in tf.range(10):
#         x += 1
#     return x
# print(tf.autograph.to_code(add_10.python_function))
# def tf__add_10(x):
#   do_return = False
#   retval_ = ag__.UndefinedReturnValue()
#   with ag__.FunctionScope('add_10', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
#
#     def get_state():
#       return ()
#
#     def set_state(_):
#       pass
#
#     def loop_body(iterates, x):
#       i = iterates
#       x += 1
#       return x,
#     x, = ag__.for_stmt(ag__.converted_call(tf.range, (10,), None, fscope), None, loop_body, get_state, set_state, (x,), ('x',), ())
#     do_return = True
#     retval_ = fscope.mark_return_value(x)
#   do_return,
#   return ag__.retval(retval_)



















