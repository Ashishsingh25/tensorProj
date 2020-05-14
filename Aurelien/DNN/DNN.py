import numpy as np
import tensorflow as tf
from tensorflow import keras
from functools import partial

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
pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)
X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds
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

### Reusing Pretrained Layers

# split a small data set (set B) containing only sandals or shirts
# def split_dataset(X, y):
#     y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
#     y_A = y[~y_5_or_6]
#     y_A[y_A > 6] -= 2 # class indices 7, 8, 9 should be moved to 5, 6, 7
#     y_B = (y[y_5_or_6] == 6).astype(np.float32) # binary classification task: is it a shirt (class 6)?
#     return ((X[~y_5_or_6], y_A),
#             (X[y_5_or_6], y_B))
#
# (X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
# (X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
# (X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
# X_train_B = X_train_B[:200]
# y_train_B = y_train_B[:200]

# print(X_train_A.shape) # main data set
# # (43986, 28, 28)
# print(X_train_B.shape) # small data set
# # (200, 28, 28)
# print(y_train_A[:30])
# # [4 0 5 7 7 7 4 4 3 4 0 1 6 3 4 3 2 6 5 3 4 5 1 3 4 2 0 6 7 1]
# print(y_train_B[:30])
# # [1. 1. 0. 0. 0. 0. 1. 1. 1. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 1. 1. 0. 1. 1. 1. 1.]

# train main model on main data set
# tf.random.set_seed(42)
# np.random.seed(42)
# model_A = keras.models.Sequential()
# model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
# for n_hidden in (300, 100, 50, 50, 50):
#     model_A.add(keras.layers.Dense(n_hidden, activation="selu"))
# model_A.add(keras.layers.Dense(8, activation="softmax"))
# model_A.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["accuracy"])
# history = model_A.fit(X_train_A, y_train_A, epochs=20, validation_data=(X_valid_A, y_valid_A))
# # loss: 0.2158 - accuracy: 0.9267 - val_loss: 0.2329 - val_accuracy: 0.9210
# model_A.save("my_model_A.h5")

# train model on small data set
# model_B = keras.models.Sequential()
# model_B.add(keras.layers.Flatten(input_shape=[28, 28]))
# for n_hidden in (300, 100, 50, 50, 50):
#     model_B.add(keras.layers.Dense(n_hidden, activation="selu"))
# model_B.add(keras.layers.Dense(1, activation="sigmoid"))
# model_B.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["accuracy"])
# history = model_B.fit(X_train_B, y_train_B, epochs=20, validation_data=(X_valid_B, y_valid_B))
# loss: 0.1229 - accuracy: 0.9900 - val_loss: 0.1450 - val_accuracy: 0.9696

# train model B using model A

# model_A = keras.models.load_model("my_model_A.h5")
# model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
# model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))
#
# model_A_clone = keras.models.clone_model(model_A) # save the original model and its wts
# model_A_clone.set_weights(model_A.get_weights())
#
# for layer in model_B_on_A.layers[:-1]: # freezing the pretrained layers for the first 4 epochs
#     layer.trainable = False
# model_B_on_A.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["accuracy"])
# history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4, validation_data=(X_valid_B, y_valid_B))
#
# for layer in model_B_on_A.layers[:-1]: # unfreezing the pretrained layers after first 4 epochs
#     layer.trainable = True
#
# model_B_on_A.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1e-3), metrics=["accuracy"])
# history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, validation_data=(X_valid_B, y_valid_B))
# # loss: 0.0496 - accuracy: 0.9950 - val_loss: 0.0705 - val_accuracy: 0.9888
#
# print(model_B.evaluate(X_test_B, y_test_B))
# # [0.12476818764209748, 0.9835]
# print(model_B_on_A.evaluate(X_test_B, y_test_B))
# # [0.06512182432413101, 0.9935]

### Learning Rate Scheduling

tf.random.set_seed(42)
np.random.seed(42)

# Power Scheduling
# lr = lr0 / (1 + steps / s)**c
# Keras uses c=1 and s = 1 / decay

# optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
#     keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
#     keras.layers.Dense(10, activation="softmax")])
# model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# n_epochs = 25
# history = model.fit(X_train_scaled, y_train, epochs=n_epochs,validation_data=(X_valid_scaled, y_valid))
# loss: 0.2070 - accuracy: 0.9290 - val_loss: 0.3222 - val_accuracy: 0.8884

# Exponential Scheduling
# lr = lr0 * 0.1**(epoch / s)

# def exponential_decay(lr0, s):
#     def exponential_decay_fn(epoch):
#         return lr0 * 0.1**(epoch / s)
#     return exponential_decay_fn
#
# exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
#
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
#     keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
#     keras.layers.Dense(10, activation="softmax")])
# model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
# n_epochs = 25
# lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
# history = model.fit(X_train_scaled, y_train, epochs=n_epochs, validation_data=(X_valid_scaled, y_valid),
#                     callbacks=[lr_scheduler])
# loss: 0.1166 - accuracy: 0.9613 - val_loss: 0.5827 - val_accuracy: 0.8886

# Performance Scheduling

# lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
#     keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
#     keras.layers.Dense(10, activation="softmax")])
# optimizer = keras.optimizers.SGD(lr=0.02, momentum=0.9)
# model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# n_epochs = 25
# history = model.fit(X_train_scaled, y_train, epochs=n_epochs, validation_data=(X_valid_scaled, y_valid),
#                     callbacks=[lr_scheduler])
# loss: 0.0447 - accuracy: 0.9853 - val_loss: 0.5021 - val_accuracy: 0.8958

# tf.keras schedulers

# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation="selu", kernel_initializer="lecun_normal"),
#     keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"),
#     keras.layers.Dense(10, activation="softmax")])
# s = 20 * len(X_train) // 32 # number of steps in 20 epochs (batch size = 32)
# learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
# optimizer = keras.optimizers.SGD(learning_rate)
# model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
# n_epochs = 25
# history = model.fit(X_train_scaled, y_train, epochs=n_epochs, validation_data=(X_valid_scaled, y_valid))
# loss: 0.2194 - accuracy: 0.9239 - val_loss: 0.3221 - val_accuracy: 0.8878

### Regularization

# l1 and l2

# layer = keras.layers.Dense(100, activation="elu",
#                            kernel_initializer="he_normal",
#                            kernel_regularizer=keras.regularizers.l2(0.01))
# or l1(0.1) for ℓ1 regularization with a factor or 0.1
# or l1_l2(0.1, 0.01) for both ℓ1 and ℓ2 regularization, with factors 0.1 and 0.01 respectively

# Instead of declaring parameters for layers repeated, better create a function using functool

# RegularizedDense = partial(keras.layers.Dense, activation="elu", kernel_initializer="he_normal",
#                            kernel_regularizer=keras.regularizers.l2(0.01))
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     RegularizedDense(300),
#     RegularizedDense(100),
#     RegularizedDense(10, activation="softmax")])
# model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
# n_epochs = 20
# history = model.fit(X_train_scaled, y_train, epochs=n_epochs, validation_data=(X_valid_scaled, y_valid))
# loss: 0.6757 - accuracy: 0.8406 - val_loss: 0.6687 - val_accuracy: 0.8452

# Dropout

# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dropout(rate=0.2),
#     keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
#     keras.layers.Dropout(rate=0.2),
#     keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
#     keras.layers.Dropout(rate=0.2),
#     keras.layers.Dense(10, activation="softmax")])
# model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
# n_epochs = 20
# history = model.fit(X_train_scaled, y_train, epochs=n_epochs, validation_data=(X_valid_scaled, y_valid))
# loss: 0.2832 - accuracy: 0.8945 - val_loss: 0.3001 - val_accuracy: 0.8944

# Max Norm

# MaxNormDense = partial(keras.layers.Dense, activation="selu", kernel_initializer="lecun_normal",
#                        kernel_constraint=keras.constraints.max_norm(1.))
#
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     MaxNormDense(300),
#     MaxNormDense(100),
#     keras.layers.Dense(10, activation="softmax")])
# model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
# n_epochs = 20
# history = model.fit(X_train_scaled, y_train, epochs=n_epochs, validation_data=(X_valid_scaled, y_valid))
# loss: 0.2913 - accuracy: 0.8905 - val_loss: 0.3201 - val_accuracy: 0.8812

