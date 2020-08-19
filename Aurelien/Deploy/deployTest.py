import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

### TensorFlow Serving

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full[..., np.newaxis].astype(np.float32) / 255.
X_test = X_test[..., np.newaxis].astype(np.float32) / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_new = X_test[:3]

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28, 1]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-2),
              metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

# print(np.round(model.predict(X_new), 2))
# [[0.   0.   0.   0.   0.   0.   0.   1.   0.   0.  ]
#  [0.   0.   0.99 0.01 0.   0.   0.   0.   0.   0.  ]
#  [0.   0.97 0.01 0.   0.   0.   0.   0.01 0.   0.  ]]

model_version = "0001"
model_name = "my_mnist_model"
model_path = os.path.join(model_name, model_version)
print(model_path)
tf.saved_model.save(model, model_path)


