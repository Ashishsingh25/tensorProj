import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

### Simple AutoEncoder (PCA)

# np.random.seed(4)
# def generate_3d_data(m, w1=0.1, w2=0.3, noise=0.1):
#     angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
#     data = np.empty((m, 3))
#     data[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
#     data[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
#     data[:, 2] = data[:, 0] * w1 + data[:, 1] * w2 + noise * np.random.randn(m)
#     return data
#
# X_train = generate_3d_data(60)
# X_train = X_train - X_train.mean(axis=0, keepdims=0)
#
# np.random.seed(42)
# tf.random.set_seed(42)
# encoder = keras.models.Sequential([keras.layers.Dense(2, input_shape=[3])])
# decoder = keras.models.Sequential([keras.layers.Dense(3, input_shape=[2])])
# autoencoder = keras.models.Sequential([encoder, decoder])
# autoencoder.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1.5))
# history = autoencoder.fit(X_train, X_train, epochs=20)
# # loss: 0.0054
# codings = encoder.predict(X_train)

# fig = plt.figure(figsize=(4,3))
# plt.plot(codings[:,0], codings[:, 1], "b.")
# plt.xlabel("$z_1$", fontsize=18)
# plt.ylabel("$z_2$", fontsize=18, rotation=0)
# plt.grid(True)
# plt.show()

### Stacked Autoencoders

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

# tf.random.set_seed(42)
# np.random.seed(42)
# stacked_encoder = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(100, activation="selu"),
#     keras.layers.Dense(30, activation="selu")])
# stacked_decoder = keras.models.Sequential([
#     keras.layers.Dense(100, activation="selu", input_shape=[30]),
#     keras.layers.Dense(28 * 28, activation="sigmoid"),
#     keras.layers.Reshape([28, 28])])
# stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])
# stacked_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.5), metrics=[rounded_accuracy])
# history = stacked_ae.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))
# loss: 0.2822 - rounded_accuracy: 0.9352 - val_loss: 0.2831 - val_rounded_accuracy: 0.9357

def plot_image(image):
    plt.imshow(image, cmap="binary")
    plt.axis("off")

def show_reconstructions(model, images=X_valid, n_images=5):
    reconstructions = model.predict(images[:n_images])
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for image_index in range(n_images):
        plt.subplot(2, n_images, 1 + image_index)
        plot_image(images[image_index])
        plt.subplot(2, n_images, 1 + n_images + image_index)
        plot_image(reconstructions[image_index])

# show_reconstructions(stacked_ae)
# plt.show()

# Visualizing MNIST in 2d

# np.random.seed(42)
#
# X_valid_compressed = stacked_encoder.predict(X_valid)
# tsne = TSNE()
# X_valid_2D = tsne.fit_transform(X_valid_compressed)
# X_valid_2D = (X_valid_2D - X_valid_2D.min()) / (X_valid_2D.max() - X_valid_2D.min())
# plt.scatter(X_valid_2D[:, 0], X_valid_2D[:, 1], c=y_valid, s=10, cmap="tab10")
# plt.axis("off")
# plt.show()

# Tying weights

# class DenseTranspose(keras.layers.Layer):
#     def __init__(self, dense, activation=None, **kwargs):
#         self.dense = dense
#         self.activation = keras.activations.get(activation)
#         super().__init__(**kwargs)
#     def build(self, batch_input_shape):
#         self.biases = self.add_weight(name="bias",
#                                       shape=[self.dense.input_shape[-1]],
#                                       initializer="zeros")
#         super().build(batch_input_shape)
#     def call(self, inputs):
#         z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
#         return self.activation(z + self.biases)
#
# keras.backend.clear_session()
# tf.random.set_seed(42)
# np.random.seed(42)
# dense_1 = keras.layers.Dense(100, activation="selu")
# dense_2 = keras.layers.Dense(30, activation="selu")
# tied_encoder = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     dense_1,
#     dense_2])
# tied_decoder = keras.models.Sequential([
#     DenseTranspose(dense_2, activation="selu"),
#     DenseTranspose(dense_1, activation="sigmoid"),
#     keras.layers.Reshape([28, 28])])
# tied_ae = keras.models.Sequential([tied_encoder, tied_decoder])
# tied_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.5), metrics=[rounded_accuracy])
# history = tied_ae.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))
# # loss: 0.2788 - rounded_accuracy: 0.9393 - val_loss: 0.2797 - val_rounded_accuracy: 0.9402
# show_reconstructions(tied_ae)
# plt.show()

### Convolutional AE

# tf.random.set_seed(42)
# np.random.seed(42)
#
# conv_encoder = keras.models.Sequential([
#     keras.layers.Reshape([28, 28, 1], input_shape=[28, 28]),
#     keras.layers.Conv2D(16, kernel_size=3, padding="SAME", activation="selu"),
#     keras.layers.MaxPool2D(pool_size=2),
#     keras.layers.Conv2D(32, kernel_size=3, padding="SAME", activation="selu"),
#     keras.layers.MaxPool2D(pool_size=2),
#     keras.layers.Conv2D(64, kernel_size=3, padding="SAME", activation="selu"),
#     keras.layers.MaxPool2D(pool_size=2)])
#
# conv_decoder = keras.models.Sequential([
#     keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="VALID", activation="selu",
#                                  input_shape=[3, 3, 64]),
#     keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding="SAME", activation="selu"),
#     keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="SAME", activation="sigmoid"),
#     keras.layers.Reshape([28, 28])])
# conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])
#
# conv_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0), metrics=[rounded_accuracy])
# history = conv_ae.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))
# # loss: 0.2584 - rounded_accuracy: 0.9605 - val_loss: 0.2591 - val_rounded_accuracy: 0.9615
# print(conv_encoder.summary(),
# conv_decoder.summary())
# # Model: "sequential"
# # _________________________________________________________________
# # Layer (type)                 Output Shape              Param #
# # =================================================================
# # reshape (Reshape)            (None, 28, 28, 1)         0
# # _________________________________________________________________
# # conv2d (Conv2D)              (None, 28, 28, 16)        160
# # _________________________________________________________________
# # max_pooling2d (MaxPooling2D) (None, 14, 14, 16)        0
# # _________________________________________________________________
# # conv2d_1 (Conv2D)            (None, 14, 14, 32)        4640
# # _________________________________________________________________
# # max_pooling2d_1 (MaxPooling2 (None, 7, 7, 32)          0
# # _________________________________________________________________
# # conv2d_2 (Conv2D)            (None, 7, 7, 64)          18496
# # _________________________________________________________________
# # max_pooling2d_2 (MaxPooling2 (None, 3, 3, 64)          0
# # =================================================================
# # Total params: 23,296
# # Trainable params: 23,296
# # Non-trainable params: 0
# # _________________________________________________________________
# # Model: "sequential_1"
# # _________________________________________________________________
# # Layer (type)                 Output Shape              Param #
# # =================================================================
# # conv2d_transpose (Conv2DTran (None, 7, 7, 32)          18464
# # _________________________________________________________________
# # conv2d_transpose_1 (Conv2DTr (None, 14, 14, 16)        4624
# # _________________________________________________________________
# # conv2d_transpose_2 (Conv2DTr (None, 28, 28, 1)         145
# # _________________________________________________________________
# # reshape_1 (Reshape)          (None, 28, 28)            0
# # =================================================================
# # Total params: 23,233
# # Trainable params: 23,233
# # Non-trainable params: 0
#
# show_reconstructions(conv_ae)
# plt.show()

### Recurrent Autoencoders

# recurrent_encoder = keras.models.Sequential([
#     keras.layers.LSTM(100, return_sequences=True, input_shape=[28, 28]),
#     keras.layers.LSTM(30)])
# recurrent_decoder = keras.models.Sequential([
#     keras.layers.RepeatVector(28, input_shape=[30]),
#     keras.layers.LSTM(100, return_sequences=True),
#     keras.layers.TimeDistributed(keras.layers.Dense(28, activation="sigmoid"))])
# recurrent_ae = keras.models.Sequential([recurrent_encoder, recurrent_decoder])
# recurrent_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(0.1), metrics=[rounded_accuracy])
# history = recurrent_ae.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))
# # loss: 0.3044 - rounded_accuracy: 0.9143 - val_loss: 0.3059 - val_rounded_accuracy: 0.9136
# show_reconstructions(recurrent_ae)
# plt.show()

### Denoising Autoencoders

# Using Gaussian noise:

# tf.random.set_seed(42)
# np.random.seed(42)
# denoising_encoder = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.GaussianNoise(0.2),
#     keras.layers.Dense(100, activation="selu"),
#     keras.layers.Dense(30, activation="selu")])
#
# denoising_decoder = keras.models.Sequential([
#     keras.layers.Dense(100, activation="selu", input_shape=[30]),
#     keras.layers.Dense(28 * 28, activation="sigmoid"),
#     keras.layers.Reshape([28, 28])])
# denoising_ae = keras.models.Sequential([denoising_encoder, denoising_decoder])
# denoising_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0), metrics=[rounded_accuracy])
# history = denoising_ae.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))
# # loss: 0.2876 - rounded_accuracy: 0.9306 - val_loss: 0.2862 - val_rounded_accuracy: 0.9350
# noise = keras.layers.GaussianNoise(0.2)
# show_reconstructions(denoising_ae, noise(X_valid, training=True))
# plt.show()

# dropout

# tf.random.set_seed(42)
# np.random.seed(42)
# dropout_encoder = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(100, activation="selu"),
#     keras.layers.Dense(30, activation="selu")])
# dropout_decoder = keras.models.Sequential([
#     keras.layers.Dense(100, activation="selu", input_shape=[30]),
#     keras.layers.Dense(28 * 28, activation="sigmoid"),
#     keras.layers.Reshape([28, 28])])
# dropout_ae = keras.models.Sequential([dropout_encoder, dropout_decoder])
# dropout_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0), metrics=[rounded_accuracy])
# history = dropout_ae.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))
# # loss: 0.2955 - rounded_accuracy: 0.9213 - val_loss: 0.2916 - val_rounded_accuracy: 0.9267
# dropout = keras.layers.Dropout(0.5)
# show_reconstructions(dropout_ae, dropout(X_valid, training=True))
# plt.show()

### Sparse Autoencoder

# using l1 regularization

# tf.random.set_seed(42)
# np.random.seed(42)
# sparse_l1_encoder = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(100, activation="selu"),
#     keras.layers.Dense(300, activation="sigmoid"),
#     keras.layers.ActivityRegularization(l1=1e-3)])
# # Alternatively, you could add activity_regularizer=keras.regularizers.l1(1e-3) to the previous layer.
# sparse_l1_decoder = keras.models.Sequential([
#     keras.layers.Dense(100, activation="selu", input_shape=[300]),
#     keras.layers.Dense(28 * 28, activation="sigmoid"),
#     keras.layers.Reshape([28, 28])])
# sparse_l1_ae = keras.models.Sequential([sparse_l1_encoder, sparse_l1_decoder])
# sparse_l1_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0), metrics=[rounded_accuracy])
# history = sparse_l1_ae.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))
# # loss: 0.3089 - rounded_accuracy: 0.9150 - val_loss: 0.3100 - val_rounded_accuracy: 0.9145
# show_reconstructions(sparse_l1_ae)
# plt.show()

# KL Divergence loss

K = keras.backend
kl_divergence = keras.losses.kullback_leibler_divergence
class KLDivergenceRegularizer(keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target
    def __call__(self, inputs):
        mean_activities = K.mean(inputs, axis=0)
        return self.weight * (
            kl_divergence(self.target, mean_activities) +
            kl_divergence(1. - self.target, 1. - mean_activities))

tf.random.set_seed(42)
np.random.seed(42)
kld_reg = KLDivergenceRegularizer(weight=0.05, target=0.1)
sparse_kl_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(300, activation="sigmoid", activity_regularizer=kld_reg)])

sparse_kl_decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[300]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])])
sparse_kl_ae = keras.models.Sequential([sparse_kl_encoder, sparse_kl_decoder])
sparse_kl_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.0), metrics=[rounded_accuracy])
history = sparse_kl_ae.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))
# loss: 0.2928 - rounded_accuracy: 0.9277 - val_loss: 0.2942 - val_rounded_accuracy: 0.9280
show_reconstructions(sparse_kl_ae)
plt.show()