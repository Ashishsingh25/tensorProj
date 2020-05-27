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

class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name="bias",
                                      shape=[self.dense.input_shape[-1]],
                                      initializer="zeros")
        super().build(batch_input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)
dense_1 = keras.layers.Dense(100, activation="selu")
dense_2 = keras.layers.Dense(30, activation="selu")
tied_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    dense_1,
    dense_2])
tied_decoder = keras.models.Sequential([
    DenseTranspose(dense_2, activation="selu"),
    DenseTranspose(dense_1, activation="sigmoid"),
    keras.layers.Reshape([28, 28])])
tied_ae = keras.models.Sequential([tied_encoder, tied_decoder])
tied_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.SGD(lr=1.5), metrics=[rounded_accuracy])
history = tied_ae.fit(X_train, X_train, epochs=20, validation_data=(X_valid, X_valid))
# loss: 0.2788 - rounded_accuracy: 0.9393 - val_loss: 0.2797 - val_rounded_accuracy: 0.9402
show_reconstructions(tied_ae)
plt.show()