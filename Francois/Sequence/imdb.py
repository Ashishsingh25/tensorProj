from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

max_features = 10000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
# 25000 train sequences
# 25000 test sequences

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
# x_train shape: (25000, 500)
# x_test shape: (25000, 500)

# simple 1D convnet

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

# model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 500, 128)          1280000
# _________________________________________________________________
# conv1d (Conv1D)              (None, 494, 32)           28704
# _________________________________________________________________
# max_pooling1d (MaxPooling1D) (None, 98, 32)            0
# _________________________________________________________________
# conv1d_1 (Conv1D)            (None, 92, 32)            7200
# _________________________________________________________________
# global_max_pooling1d (Global (None, 32)                0
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 1,315,937
# Trainable params: 1,315,937
# Non-trainable params: 0

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
model.save('conv1d.h5')
# Epoch 1/10
# 157/157 [==============================] - 6s 39ms/step - loss: 0.9200 - acc: 0.5056 - val_loss: 0.6906 - val_acc: 0.5458
# Epoch 2/10
# 157/157 [==============================] - 6s 37ms/step - loss: 0.6770 - acc: 0.6187 - val_loss: 0.6670 - val_acc: 0.6496
# Epoch 3/10
# 157/157 [==============================] - 6s 37ms/step - loss: 0.6305 - acc: 0.7406 - val_loss: 0.6086 - val_acc: 0.7428
# Epoch 4/10
# 157/157 [==============================] - 6s 37ms/step - loss: 0.5379 - acc: 0.8026 - val_loss: 0.5068 - val_acc: 0.7904
# Epoch 5/10
# 157/157 [==============================] - 6s 37ms/step - loss: 0.4189 - acc: 0.8477 - val_loss: 0.4246 - val_acc: 0.8324
# Epoch 6/10
# 157/157 [==============================] - 6s 37ms/step - loss: 0.3543 - acc: 0.8716 - val_loss: 0.3963 - val_acc: 0.8486
# Epoch 7/10
# 157/157 [==============================] - 6s 37ms/step - loss: 0.3187 - acc: 0.8864 - val_loss: 0.4005 - val_acc: 0.8520
# Epoch 8/10
# 157/157 [==============================] - 6s 37ms/step - loss: 0.2863 - acc: 0.9003 - val_loss: 0.3963 - val_acc: 0.8606
# Epoch 9/10
# 157/157 [==============================] - 6s 37ms/step - loss: 0.2621 - acc: 0.9089 - val_loss: 0.4212 - val_acc: 0.8658
# Epoch 10/10
# 157/157 [==============================] - 6s 37ms/step - loss: 0.2394 - acc: 0.9186 - val_loss: 0.4179 - val_acc: 0.8700

# Plotting

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('conv1d.png', bbox_inches='tight')

plt.show()