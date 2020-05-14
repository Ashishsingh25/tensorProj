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

def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn
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

precision = keras.metrics.Precision()
precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
print(precision.result())
# tf.Tensor(0.8, shape=(), dtype=float32)
precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0])
print(precision.result())
# tf.Tensor(0.5, shape=(), dtype=float32)

class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs) # handles base args (e.g., dtype)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold) # TODO: investigate why this fails
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")
    def huber_fn(self, y_true, y_pred): # workaround
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
    def result(self):
        return self.total / self.count
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal", input_shape=input_shape),
    keras.layers.Dense(1)])
model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[HuberMetric(2.0)])
model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)
# loss: 0.2564 - huber_metric: 0.2564