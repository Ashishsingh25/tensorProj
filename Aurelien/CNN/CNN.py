import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image
import tensorflow as tf
from tensorflow import keras
from functools import partial

def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")

def plot_color_image(image):
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")

### Load sample images
# china = load_sample_image("china.jpg") / 255
# flower = load_sample_image("flower.jpg") / 255
# images = np.array([china, flower])
# batch_size, height, width, channels = images.shape
# print(images.shape)
# (2, 427, 640, 3)
# Create 2 filters
# filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
# filters[:, 3, :, 0] = 1  # vertical line
# filters[3, :, :, 1] = 1  # horizontal line
# print(filters[:, :, 0, 0]) # vertical line
# [[0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0.]]
# outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
# for image_index in (0, 1):
#     for feature_map_index in (0, 1):
#         plt.subplot(2, 2, image_index * 2 + feature_map_index + 1)
#         plot_image(outputs[image_index, :, :, feature_map_index])
# plt.show()

### Fashion MNIST

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation='relu', padding="SAME")
#
# model = keras.models.Sequential([
#     DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
#     keras.layers.MaxPooling2D(pool_size=2),
#     DefaultConv2D(filters=128),
#     DefaultConv2D(filters=128),
#     keras.layers.MaxPooling2D(pool_size=2),
#     DefaultConv2D(filters=256),
#     DefaultConv2D(filters=256),
#     keras.layers.MaxPooling2D(pool_size=2),
#     keras.layers.Flatten(),
#     keras.layers.Dense(units=128, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(units=64, activation='relu'),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(units=10, activation='softmax')])
# model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
# loss: 0.2484 - accuracy: 0.9150 - val_loss: 0.3016 - val_accuracy: 0.8978
# score = model.evaluate(X_test, y_test)
# loss: 0.3146 - accuracy: 0.8938
# X_new = X_test[:5] # pretend we have new images
# y_pred = model.predict(X_new)
# print(y_test[:5])
# [9 2 1 1 6]
# print(y_pred)
# [[3.13858542e-26 2.04355014e-31 5.37001329e-29 8.82340655e-30
#   1.46338796e-30 6.19055198e-12 1.19482419e-23 1.34754655e-05
#   1.76284005e-24 9.99986529e-01]
#  [3.20721172e-10 1.04412919e-24 9.99925375e-01 4.85089234e-18
#   4.80415792e-07 6.02609190e-37 7.41944095e-05 0.00000000e+00
#   2.22962447e-20 0.00000000e+00]
#  [0.00000000e+00 1.00000000e+00 0.00000000e+00 2.84920059e-31
#   0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#   0.00000000e+00 0.00000000e+00]
#  [0.00000000e+00 1.00000000e+00 0.00000000e+00 5.70015137e-32
#   0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00
#   0.00000000e+00 0.00000000e+00]
#  [3.05447029e-04 1.12092562e-12 1.75268346e-04 1.47005025e-07
#   1.84677751e-03 2.13266667e-17 9.97672379e-01 6.33715281e-20
#   2.03335040e-11 7.32800839e-22]]

### ResNet 34

# DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
#                         padding="SAME", use_bias=False)
#
# class ResidualUnit(keras.layers.Layer):
#     def __init__(self, filters, strides=1, activation="relu", **kwargs):
#         super().__init__(**kwargs)
#         self.activation = keras.activations.get(activation)
#         self.main_layers = [
#             DefaultConv2D(filters, strides=strides),
#             keras.layers.BatchNormalization(),
#             self.activation,
#             DefaultConv2D(filters),
#             keras.layers.BatchNormalization()]
#         self.skip_layers = []
#         if strides > 1:
#             self.skip_layers = [
#                 DefaultConv2D(filters, kernel_size=1, strides=strides),
#                 keras.layers.BatchNormalization()]
#
#     def call(self, inputs):
#         Z = inputs
#         for layer in self.main_layers:
#             Z = layer(Z)
#         skip_Z = inputs
#         for layer in self.skip_layers:
#             skip_Z = layer(skip_Z)
#         return self.activation(Z + skip_Z)
# model = keras.models.Sequential()
# model.add(DefaultConv2D(64, kernel_size=7, strides=2,
#                         input_shape=[224, 224, 3]))
# model.add(keras.layers.BatchNormalization())
# model.add(keras.layers.Activation("relu"))
# model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
# prev_filters = 64
# for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
#     strides = 1 if filters == prev_filters else 2
#     model.add(ResidualUnit(filters, strides=strides))
#     prev_filters = filters
# model.add(keras.layers.GlobalAvgPool2D())
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(10, activation="softmax"))
# print(model.summary())
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 112, 112, 64)      9408
# _________________________________________________________________
# batch_normalization (BatchNo (None, 112, 112, 64)      256
# _________________________________________________________________
# activation (Activation)      (None, 112, 112, 64)      0
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 56, 56, 64)        0
# _________________________________________________________________
# residual_unit (ResidualUnit) (None, 56, 56, 64)        74240
# _________________________________________________________________
# residual_unit_1 (ResidualUni (None, 56, 56, 64)        74240
# _________________________________________________________________
# residual_unit_2 (ResidualUni (None, 56, 56, 64)        74240
# _________________________________________________________________
# residual_unit_3 (ResidualUni (None, 28, 28, 128)       230912
# _________________________________________________________________
# residual_unit_4 (ResidualUni (None, 28, 28, 128)       295936
# _________________________________________________________________
# residual_unit_5 (ResidualUni (None, 28, 28, 128)       295936
# _________________________________________________________________
# residual_unit_6 (ResidualUni (None, 28, 28, 128)       295936
# _________________________________________________________________
# residual_unit_7 (ResidualUni (None, 14, 14, 256)       920576
# _________________________________________________________________
# residual_unit_8 (ResidualUni (None, 14, 14, 256)       1181696
# _________________________________________________________________
# residual_unit_9 (ResidualUni (None, 14, 14, 256)       1181696
# _________________________________________________________________
# residual_unit_10 (ResidualUn (None, 14, 14, 256)       1181696
# _________________________________________________________________
# residual_unit_11 (ResidualUn (None, 14, 14, 256)       1181696
# _________________________________________________________________
# residual_unit_12 (ResidualUn (None, 14, 14, 256)       1181696
# _________________________________________________________________
# residual_unit_13 (ResidualUn (None, 7, 7, 512)         3676160
# _________________________________________________________________
# residual_unit_14 (ResidualUn (None, 7, 7, 512)         4722688
# _________________________________________________________________
# residual_unit_15 (ResidualUn (None, 7, 7, 512)         4722688
# _________________________________________________________________
# global_average_pooling2d (Gl (None, 512)               0
# _________________________________________________________________
# flatten (Flatten)            (None, 512)               0
# _________________________________________________________________
# dense (Dense)                (None, 10)                5130
# =================================================================
# Total params: 21,306,826
# Trainable params: 21,289,802
# Non-trainable params: 17,024
# _________________________________________________________________













