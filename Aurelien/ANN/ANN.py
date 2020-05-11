import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


### Perceptron
# iris = load_iris()
# X = iris.data[:, (2, 3)] # petal length, petal width
# y = (iris.target == 0).astype(np.int) # Iris Setosa?
# per_clf = Perceptron()
# per_clf.fit(X, y)
# y_pred = per_clf.predict([[2, 0.5]])
# print(y_pred)
# [0]

### Fashion MNIST - Classifier
# fashion_mnist = keras.datasets.fashion_mnist
# (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
# print(X_train_full.shape, X_train_full.dtype)
# # (60000, 28, 28) uint8
#
# X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
# y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# X_test = X_test / 255.
#
# plt.imshow(X_train[0], cmap="binary")
# plt.axis('off')
# plt.show()
#
# class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
#                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
# print(y_train, class_names[y_train[0]])
# # [4 0 7 ... 3 0 5] Coat
# print(X_valid.shape, X_train.shape)
# # (5000, 28, 28) (55000, 28, 28)
# keras.backend.clear_session()
# np.random.seed(42)
# tf.random.set_seed(42)
# model = keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation="relu"),
#     keras.layers.Dense(100, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])
# print(model.layers)
# # [<tensorflow.python.keras.layers.core.Flatten object at 0x00000256E2DC34C8>,
# # <tensorflow.python.keras.layers.core.Dense object at 0x00000256DA1AC848>,
# # <tensorflow.python.keras.layers.core.Dense object at 0x00000256D9E22908>,
# # <tensorflow.python.keras.layers.core.Dense object at 0x00000256E3087A08>]
# print(model.summary())
# # Model: "sequential"
# # _________________________________________________________________
# # Layer (type)                 Output Shape              Param #
# # =================================================================
# # flatten (Flatten)            (None, 784)               0
# # _________________________________________________________________
# # dense (Dense)                (None, 300)               235500
# # _________________________________________________________________
# # dense_1 (Dense)              (None, 100)               30100
# # _________________________________________________________________
# # dense_2 (Dense)              (None, 10)                1010
# # =================================================================
# # Total params: 266,610
# # Trainable params: 266,610
# # Non-trainable params: 0
# # _________________________________________________________________
# hidden1 = model.layers[1]
# print(hidden1.name)
# # dense
# weights, biases = hidden1.get_weights()
# print(weights.shape)
# # (784, 300)
# print(biases)
# # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
# #  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
# history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
# # loss: 0.2256 - accuracy: 0.9192 - val_loss: 0.3016 - val_accuracy: 0.8910
# print(history.params)
# # {'batch_size': 32, 'epochs': 30, 'steps': 1719, 'samples': 55000, 'verbose': 0, 'do_validation': True,
# # 'metrics': ['loss', 'accuracy', 'val_loss', 'val_accuracy']}
# print(history.epoch)
# # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
# print(history.history.keys())
# # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
#
# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.show()
# print(model.evaluate(X_test, y_test))
# # [0.33459347892999647, 0.8784]
# X_new = X_test[:3]
# y_proba = model.predict(X_new)
# print(y_proba.round(2))
# # [[0.   0.   0.   0.   0.   0.   0.   0.01 0.   0.98]
# #  [0.   0.   0.99 0.   0.   0.   0.   0.   0.   0.  ]
# #  [0.   1.   0.   0.   0.   0.   0.   0.   0.   0.  ]]
#
# y_pred = model.predict_classes(X_new)
# print(y_pred)
# # [9 2 1]
# print(np.array(class_names)[y_pred])
# # ['Ankle boot' 'Pullover' 'Trouser']
# y_new = y_test[:3]
# print(y_new)
# # [9 2 1]
# plt.figure(figsize=(7.2, 2.4))
# for index, image in enumerate(X_new):
#     plt.subplot(1, 3, index + 1)
#     plt.imshow(image, cmap="binary", interpolation="nearest")
#     plt.axis('off')
#     plt.title(class_names[y_test[index]], fontsize=12)
# plt.subplots_adjust(wspace=0.2, hspace=0.5)
# plt.show()

### Cali Housing - Regress

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# print(X_train.shape)
# (11610, 8)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
# np.random.seed(42)
# tf.random.set_seed(42)

# simple model

# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
#     keras.layers.Dense(1)
# ])
# model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
# history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
# # loss: 0.4281 - val_loss: 0.3981
# mse_test = model.evaluate(X_test, y_test)
# print(mse_test)
# # 0.4217712501222773
X_new = X_test[:3]
# y_pred = model.predict(X_new)
# print(y_pred)
# # [[0.37310085]
# #  [1.679079  ]
# #  [3.0817142 ]]
# print(y_test[:3])
# # [0.477   0.458   5.00001]
# plt.plot(pd.DataFrame(history.history))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.show()

# passing the input and hidden layer to the output layer

# input_ = keras.layers.Input(shape=X_train.shape[1:])
# hidden1 = keras.layers.Dense(30, activation="relu")(input_)
# hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# concat = keras.layers.concatenate([input_, hidden2])
# output = keras.layers.Dense(1)(concat)
# model = keras.models.Model(inputs=[input_], outputs=[output])
# print(model.summary())
# # Model: "model"
# # __________________________________________________________________________________________________
# # Layer (type)                    Output Shape         Param #     Connected to
# # ==================================================================================================
# # input_1 (InputLayer)            [(None, 8)]          0
# # __________________________________________________________________________________________________
# # dense (Dense)                   (None, 30)           270         input_1[0][0]
# # __________________________________________________________________________________________________
# # dense_1 (Dense)                 (None, 30)           930         dense[0][0]
# # __________________________________________________________________________________________________
# # concatenate (Concatenate)       (None, 38)           0           input_1[0][0]
# #                                                                  dense_1[0][0]
# # __________________________________________________________________________________________________
# # dense_2 (Dense)                 (None, 1)            39          concatenate[0][0]
# # ==================================================================================================
# # Total params: 1,239
# # Trainable params: 1,239
# # Non-trainable params: 0
# # __________________________________________________________________________________________________
# model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
# history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
# # loss: 0.4096 - val_loss: 0.3923
# mse_test = model.evaluate(X_test, y_test)
# print(mse_test)
# # 0.4042230805685354
# y_pred = model.predict(X_new)
# print(y_pred)
# # [[0.4725365]
# #  [1.8545787]
# #  [3.341889 ]]

# passing the input and hidden layer to the output layer
# different sets of features are passed in hidden and passed directly to the output layer

# input_A = keras.layers.Input(shape=[5], name="wide_input")
# input_B = keras.layers.Input(shape=[6], name="deep_input")
# hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
# hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# concat = keras.layers.concatenate([input_A, hidden2])
# output = keras.layers.Dense(1, name="output")(concat)
# model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
# model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
#
# X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
# X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
# X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
# X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]
#
# history = model.fit((X_train_A, X_train_B), y_train, epochs=20, validation_data=((X_valid_A, X_valid_B), y_valid))
# # loss: 0.4259 - val_loss: 0.3976
# mse_test = model.evaluate((X_test_A, X_test_B), y_test)
# print(mse_test)
# # 0.42019958080247394
# y_pred = model.predict((X_new_A, X_new_B))
# print(y_pred)
# # [[0.30367878]
# #  [1.9546494 ]
# #  [3.4302533 ]]

# auxiliary output for regularization

# input_A = keras.layers.Input(shape=[5], name="wide_input")
# input_B = keras.layers.Input(shape=[6], name="deep_input")
# hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
# hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# concat = keras.layers.concatenate([input_A, hidden2])
# output = keras.layers.Dense(1, name="main_output")(concat)
# aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
# model = keras.models.Model(inputs=[input_A, input_B], outputs=[output, aux_output])
# model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))
# history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
#                     validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))
# # loss: 0.4753 - main_output_loss: 0.4245 - aux_output_loss: 0.9320 -
# # val_loss: 0.4598 - val_main_output_loss: 0.3968 - val_aux_output_loss: 1.0257
# total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])
# print(total_loss, main_loss, aux_loss)
# # 0.4656009970709335 0.41649783 0.9110531
# y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])
# print(y_pred_main)
# # [[0.26558056]
# #  [1.9819798 ]
# #  [3.3208997 ]]
# print(y_pred_aux)
# # [[0.9548485]
# #  [1.9263171]
# #  [2.4976127]]

# Subclassing API

# class WideAndDeepModel(keras.models.Model):
#     def __init__(self, units=30, activation="relu", **kwargs):
#         super().__init__(**kwargs)
#         self.hidden1 = keras.layers.Dense(units, activation=activation)
#         self.hidden2 = keras.layers.Dense(units, activation=activation)
#         self.main_output = keras.layers.Dense(1)
#         self.aux_output = keras.layers.Dense(1)
#
#     def call(self, inputs):
#         input_A, input_B = inputs
#         hidden1 = self.hidden1(input_B)
#         hidden2 = self.hidden2(hidden1)
#         concat = keras.layers.concatenate([input_A, hidden2])
#         main_output = self.main_output(concat)
#         aux_output = self.aux_output(hidden2)
#         return main_output, aux_output
#
#
# model = WideAndDeepModel(30, activation="relu")
# model.compile(loss="mse", loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))
# history = model.fit((X_train_A, X_train_B), (y_train, y_train), epochs=10,
#                     validation_data=((X_valid_A, X_valid_B), (y_valid, y_valid)))
# # loss: 0.5368 - output_1_loss: 0.4671 - output_2_loss: 1.1631 -
# # val_loss: 0.6954 - val_output_1_loss: 0.4467 - val_output_2_loss: 2.9307
# total_loss, main_loss, aux_loss = model.evaluate((X_test_A, X_test_B), (y_test, y_test))
# print(total_loss, main_loss, aux_loss)
# # 0.5222893047702405 0.4552024 1.1322285
# y_pred_main, y_pred_aux = model.predict((X_new_A, X_new_B))
# print(y_pred_main, y_pred_aux)
# # [[0.31836504]
# #  [1.6945157 ]
# #  [3.015293  ]]
# # [[1.0351286]
# #  [1.6754843]
# #  [2.2743278]]

# Saving Loading

# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation="relu", input_shape=[8]),
#     keras.layers.Dense(30, activation="relu"),
#     keras.layers.Dense(1)
# ])
# model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
# # loss: 0.4476 - val_loss: 0.4185
# mse_test = model.evaluate(X_test, y_test)
# print(mse_test)
# # 0.43759363182755406
# model.save("my_keras_model.h5")
# model = keras.models.load_model("my_keras_model.h5")
# print(model.predict(X_new))
# # [[0.55155903]
# #  [1.6555369 ]
# #  [3.0014236 ]]
# model.save_weights("my_keras_weights.ckpt")
# model.load_weights("my_keras_weights.ckpt")
# print(model.predict(X_new))
# # [[0.55155903]
# #  [1.6555369 ]
# #  [3.0014236 ]]

# Callback

# keras.backend.clear_session()
# np.random.seed(42)
# tf.random.set_seed(42)
#
# model = keras.models.Sequential([
#     keras.layers.Dense(30, activation="relu", input_shape=[8]),
#     keras.layers.Dense(30, activation="relu"),
#     keras.layers.Dense(1)])

# model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
# history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb])
# model = keras.models.load_model("my_keras_model.h5") # rollback to best model
# mse_test = model.evaluate(X_test, y_test)
# print(mse_test)
# 0.43759363182755406

# Callback with Early Stopping

# model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
# history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
#                     callbacks=[checkpoint_cb, early_stopping_cb])
# # loss: 0.3295 - val_loss: 0.3205
# mse_test = model.evaluate(X_test, y_test)
# print(mse_test)
# # 0.32932399090870407

### Fine-Tuning Hyperparameters

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
# keras_reg.fit(X_train, y_train, epochs=100,
#               validation_data=(X_valid, y_valid),
#               callbacks=[keras.callbacks.EarlyStopping(patience=10)])
# # 0.3442 - val_loss: 0.4654
# mse_test = keras_reg.score(X_test, y_test)
# print(mse_test)
# # -0.3472500825806182
# y_pred = keras_reg.predict(X_new)
# print(y_pred)
# # [0.6616513 1.650584  4.104437 ]

# using RandomizedSearchCV to compare sets of hyperparameter
param_distribs = {
    "n_hidden": np.arange(1, 3),
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=0)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[checkpoint_cb])
# rnd_search_cv = keras.models.load_model("my_keras_model.h5")
print(rnd_search_cv.best_params_)
print(rnd_search_cv.best_score_)
print(rnd_search_cv.best_estimator_)
print(rnd_search_cv.score(X_test, y_test))
model = rnd_search_cv.best_estimator_.model
print(model)
print(model.evaluate(X_test, y_test))
