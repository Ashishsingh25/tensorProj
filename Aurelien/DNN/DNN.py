import numpy as np
import tensorflow as tf
from tensorflow import keras

# print([name for name in dir(keras.initializers) if not name.startswith("_")])
# ['Constant', 'GlorotNormal', 'GlorotUniform', 'Identity', 'Initializer', 'Ones', 'Orthogonal', 'RandomNormal',
#  'RandomUniform', 'TruncatedNormal', 'VarianceScaling', 'Zeros', 'constant', 'deserialize', 'get', 'glorot_normal',
#  'glorot_uniform', 'he_normal', 'he_uniform', 'identity', 'lecun_normal', 'lecun_uniform', 'ones', 'orthogonal',
#  'serialize', 'zeros']

# print([m for m in dir(keras.activations) if not m.startswith("_")])
# ['deserialize', 'elu', 'exponential', 'get', 'hard_sigmoid', 'linear', 'relu', 'selu', 'serialize', 'sigmoid',
#  'softmax', 'softplus', 'softsign', 'tanh']

# print([m for m in dir(keras.layers) if "relu" in m.lower()])
# ['LeakyReLU', 'PReLU', 'ReLU', 'ThresholdedReLU']

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

### Leaky ReLU
# tf.random.set_seed(42)
# np.random.seed(42)
#
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, kernel_initializer="he_normal"),
#     keras.layers.LeakyReLU(),
#     keras.layers.Dense(100, kernel_initializer="he_normal"),
#     keras.layers.LeakyReLU(),
#     keras.layers.Dense(10, activation="softmax")])
# model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["accuracy"])
# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
# loss: 0.4923 - accuracy: 0.8333 - val_loss: 0.4816 - val_accuracy: 0.8396

### PReLU
# tf.random.set_seed(42)
# np.random.seed(42)
#
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, kernel_initializer="he_normal"),
#     keras.layers.PReLU(),
#     keras.layers.Dense(100, kernel_initializer="he_normal"),
#     keras.layers.PReLU(),
#     keras.layers.Dense(10, activation="softmax")])
# model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["accuracy"])
# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
# loss: 0.4946 - accuracy: 0.8321 - val_loss: 0.4839 - val_accuracy: 0.8378

### SELU with 100 layers

# np.random.seed(42)
# tf.random.set_seed(42)
# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28, 28]))
# model.add(keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"))
# for layer in range(99):
#     model.add(keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"))
# model.add(keras.layers.Dense(10, activation="softmax"))
# model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["accuracy"])
#
# pixel_means = X_train.mean(axis=0, keepdims=True)
# pixel_stds = X_train.std(axis=0, keepdims=True)
# X_train_scaled = (X_train - pixel_means) / pixel_stds
# X_valid_scaled = (X_valid - pixel_means) / pixel_stds
# X_test_scaled = (X_test - pixel_means) / pixel_stds
#
# history = model.fit(X_train_scaled, y_train, epochs=5, validation_data=(X_valid_scaled, y_valid))
# loss: 0.5523 - accuracy: 0.8069 - val_loss: 0.5349 - val_accuracy: 0.8040

### ReLU with 100 layers

# np.random.seed(42)
# tf.random.set_seed(42)
# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28, 28]))
# model.add(keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"))
# for layer in range(99):
#     model.add(keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"))
# model.add(keras.layers.Dense(10, activation="softmax"))
# model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["accuracy"])
# history = model.fit(X_train_scaled, y_train, epochs=5, validation_data=(X_valid_scaled, y_valid))
# vanishing/exploding gradients problem
# loss: 0.7260 - accuracy: 0.7067 - val_loss: 0.6577 - val_accuracy: 0.7200

# Batch Normalization

# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(300, activation="relu"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(100, activation="relu"),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(10, activation="softmax")])
# print(model.summary())
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# flatten (Flatten)            (None, 784)               0
# _________________________________________________________________
# batch_normalization (BatchNo (None, 784)               3136
# _________________________________________________________________
# dense (Dense)                (None, 300)               235500
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 300)               1200
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               30100
# _________________________________________________________________
# batch_normalization_2 (Batch (None, 100)               400
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1010
# =================================================================
# Total params: 271,346
# Trainable params: 268,978
# Non-trainable params: 2,368
# _________________________________________________________________

# model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["accuracy"])
# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
# loss: 0.3945 - accuracy: 0.8609 - val_loss: 0.3641 - val_accuracy: 0.8746

# BN before the activation function

# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(300, use_bias=False),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation("relu"),
#     keras.layers.Dense(100, use_bias=False),
#     keras.layers.BatchNormalization(),
#     keras.layers.Activation("relu"),
#     keras.layers.Dense(10, activation="softmax")])
# model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["accuracy"])
# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
# loss: 0.4302 - accuracy: 0.8517 - val_loss: 0.3794 - val_accuracy: 0.8690