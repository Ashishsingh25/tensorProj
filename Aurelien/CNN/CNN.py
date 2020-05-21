import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
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

# (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
# X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
# y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
#
# X_mean = X_train.mean(axis=0, keepdims=True)
# X_std = X_train.std(axis=0, keepdims=True) + 1e-7
# X_train = (X_train - X_mean) / X_std
# X_valid = (X_valid - X_mean) / X_std
# X_test = (X_test - X_mean) / X_std
#
# X_train = X_train[..., np.newaxis]
# X_valid = X_valid[..., np.newaxis]
# X_test = X_test[..., np.newaxis]

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

### Pretrained Model - ResNet 50

# model = keras.applications.resnet50.ResNet50(weights="imagenet")
# china_box = [0, 0.03, 1, 0.68]
# flower_box = [0.19, 0.26, 0.86, 0.7]
# images_resized = tf.image.crop_and_resize(images, [china_box, flower_box], [0, 1], [224, 224])
# tf.image.crop_and_resize(image, boxes, box_indices, CROP_SIZE)
# plot_color_image(images_resized[0])
# plt.show()
# plot_color_image(images_resized[1])
# plt.show()
# inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)
# Y_proba = model.predict(inputs)
# print(Y_proba.shape)
# # (2, 1000)
# top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
# for image_index in range(len(images)):
#     print("Image #{}".format(image_index))
#     for class_id, name, y_proba in top_K[image_index]:
#         print("  {} - {:12s} {:.2f}%".format(class_id, name, y_proba * 100))
#     print()
# Image #0
#   n03877845 - palace       43.39%
#   n02825657 - bell_cote    43.07%
#   n03781244 - monastery    11.70%
#
# Image #1
#   n04522168 - vase         53.96%
#   n07930864 - cup          9.52%
#   n11939491 - daisy        4.97%

### Transfer Learning - xception

dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
# print(info.splits)
# # {'train': <tfds.core.SplitInfo num_examples=3670>}
# print(info.splits["train"])
# # <tfds.core.SplitInfo num_examples=3670>
class_names = info.features["label"].names
# print(class_names)
# # ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
n_classes = info.features["label"].num_classes
# print(n_classes)
# # 5
dataset_size = info.splits["train"].num_examples
# print(dataset_size)
# # 3670
test_set_raw, valid_set_raw, train_set_raw = tfds.load("tf_flowers",
                                                       split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
                                                       as_supervised=True)
# plt.figure(figsize=(12, 10))
# index = 0
# for image, label in train_set_raw.take(9):
#     index += 1
#     plt.subplot(3, 3, index)
#     plt.imshow(image)
#     plt.title("Class: {}".format(class_names[label]))
#     plt.axis("off")
# plt.show()

# preprocessing
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label
batch_size = 32
train_set = train_set_raw.shuffle(1000).repeat()
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)
# plt.figure(figsize=(12, 12))
# for X_batch, y_batch in train_set.take(1):
#     for index in range(9):
#         plt.subplot(3, 3, index + 1)
#         plt.imshow(X_batch[index] / 2 + 0.5)
#         plt.title("Class: {}".format(class_names[y_batch[index]]))
#         plt.axis("off")
#
# plt.show()

# Loading only bottom layers of Xception model
base_model = keras.applications.xception.Xception(weights="imagenet",include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)
for index, layer in enumerate(base_model.layers):
    print(index, layer.name)
# 0 input_1
# 1 block1_conv1
# 2 block1_conv1_bn
# 3 block1_conv1_act
# 4 block1_conv2
# 5 block1_conv2_bn
# 6 block1_conv2_act
# 7 block2_sepconv1
# 8 block2_sepconv1_bn
# 9 block2_sepconv2_act
# 10 block2_sepconv2
# 11 block2_sepconv2_bn
# 12 conv2d
# 13 block2_pool
# 14 batch_normalization
# 15 add
# 16 block3_sepconv1_act
# 17 block3_sepconv1
# 18 block3_sepconv1_bn
# 19 block3_sepconv2_act
# 20 block3_sepconv2
# 21 block3_sepconv2_bn
# 22 conv2d_1
# 23 block3_pool
# 24 batch_normalization_1
# 25 add_1
# 26 block4_sepconv1_act
# 27 block4_sepconv1
# 28 block4_sepconv1_bn
# 29 block4_sepconv2_act
# 30 block4_sepconv2
# 31 block4_sepconv2_bn
# 32 conv2d_2
# 33 block4_pool
# 34 batch_normalization_2
# 35 add_2
# 36 block5_sepconv1_act
# 37 block5_sepconv1
# 38 block5_sepconv1_bn
# 39 block5_sepconv2_act
# 40 block5_sepconv2
# 41 block5_sepconv2_bn
# 42 block5_sepconv3_act
# 43 block5_sepconv3
# 44 block5_sepconv3_bn
# 45 add_3
# 46 block6_sepconv1_act
# 47 block6_sepconv1
# 48 block6_sepconv1_bn
# 49 block6_sepconv2_act
# 50 block6_sepconv2
# 51 block6_sepconv2_bn
# 52 block6_sepconv3_act
# 53 block6_sepconv3
# 54 block6_sepconv3_bn
# 55 add_4
# 56 block7_sepconv1_act
# 57 block7_sepconv1
# 58 block7_sepconv1_bn
# 59 block7_sepconv2_act
# 60 block7_sepconv2
# 61 block7_sepconv2_bn
# 62 block7_sepconv3_act
# 63 block7_sepconv3
# 64 block7_sepconv3_bn
# 65 add_5
# 66 block8_sepconv1_act
# 67 block8_sepconv1
# 68 block8_sepconv1_bn
# 69 block8_sepconv2_act
# 70 block8_sepconv2
# 71 block8_sepconv2_bn
# 72 block8_sepconv3_act
# 73 block8_sepconv3
# 74 block8_sepconv3_bn
# 75 add_6
# 76 block9_sepconv1_act
# 77 block9_sepconv1
# 78 block9_sepconv1_bn
# 79 block9_sepconv2_act
# 80 block9_sepconv2
# 81 block9_sepconv2_bn
# 82 block9_sepconv3_act
# 83 block9_sepconv3
# 84 block9_sepconv3_bn
# 85 add_7
# 86 block10_sepconv1_act
# 87 block10_sepconv1
# 88 block10_sepconv1_bn
# 89 block10_sepconv2_act
# 90 block10_sepconv2
# 91 block10_sepconv2_bn
# 92 block10_sepconv3_act
# 93 block10_sepconv3
# 94 block10_sepconv3_bn
# 95 add_8
# 96 block11_sepconv1_act
# 97 block11_sepconv1
# 98 block11_sepconv1_bn
# 99 block11_sepconv2_act
# 100 block11_sepconv2
# 101 block11_sepconv2_bn
# 102 block11_sepconv3_act
# 103 block11_sepconv3
# 104 block11_sepconv3_bn
# 105 add_9
# 106 block12_sepconv1_act
# 107 block12_sepconv1
# 108 block12_sepconv1_bn
# 109 block12_sepconv2_act
# 110 block12_sepconv2
# 111 block12_sepconv2_bn
# 112 block12_sepconv3_act
# 113 block12_sepconv3
# 114 block12_sepconv3_bn
# 115 add_10
# 116 block13_sepconv1_act
# 117 block13_sepconv1
# 118 block13_sepconv1_bn
# 119 block13_sepconv2_act
# 120 block13_sepconv2
# 121 block13_sepconv2_bn
# 122 conv2d_3
# 123 block13_pool
# 124 batch_normalization_3
# 125 add_11
# 126 block14_sepconv1
# 127 block14_sepconv1_bn
# 128 block14_sepconv1_act
# 129 block14_sepconv2
# 130 block14_sepconv2_bn
# 131 block14_sepconv2_act
# freezing the base layers at beginning of training
for layer in base_model.layers:
    layer.trainable = False

optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.15 * dataset_size / batch_size),
                    epochs=5)
# loss: 0.0687 - accuracy: 0.9738 - val_loss: 0.6433 - val_accuracy: 0.8695
# unfreezing to train all weights in all layers

for layer in base_model.layers:
    layer.trainable = True

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9,
                                 nesterov=True, decay=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
history = model.fit(train_set,
                    steps_per_epoch=int(0.75 * dataset_size / batch_size),
                    validation_data=valid_set,
                    validation_steps=int(0.15 * dataset_size / batch_size),
                    epochs=40)
# loss: 0.0019 - accuracy: 0.9996 - val_loss: 0.3515 - val_accuracy: 0.9320