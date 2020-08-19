from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

max_features = 10000
maxlen = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features)

x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# LSTM using reversed sequences

# model = Sequential()
# model.add(layers.Embedding(max_features, 128))
# model.add(layers.LSTM(32))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=128,
#                     validation_split=0.2)
# model.save('LSTMReversed.h5')

# bidirectional LSTM

model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
model.save('LSTMBidirectional.h5')
# Epoch 1/10
# 157/157 [==============================] - 8s 50ms/step - loss: 0.5591 - acc: 0.7246 - val_loss: 0.3869 - val_acc: 0.8558
# Epoch 2/10
# 157/157 [==============================] - 7s 46ms/step - loss: 0.3234 - acc: 0.8775 - val_loss: 0.4769 - val_acc: 0.8238
# Epoch 3/10
# 157/157 [==============================] - 7s 45ms/step - loss: 0.2647 - acc: 0.9026 - val_loss: 0.3291 - val_acc: 0.8652
# Epoch 4/10
# 157/157 [==============================] - 7s 45ms/step - loss: 0.2264 - acc: 0.9180 - val_loss: 0.3012 - val_acc: 0.8812
# Epoch 5/10
# 157/157 [==============================] - 7s 45ms/step - loss: 0.2029 - acc: 0.9276 - val_loss: 0.5222 - val_acc: 0.8400
# Epoch 6/10
# 157/157 [==============================] - 7s 45ms/step - loss: 0.1768 - acc: 0.9394 - val_loss: 0.3354 - val_acc: 0.8858
# Epoch 7/10
# 157/157 [==============================] - 7s 45ms/step - loss: 0.1686 - acc: 0.9420 - val_loss: 0.3264 - val_acc: 0.8650
# Epoch 8/10
# 157/157 [==============================] - 7s 45ms/step - loss: 0.1436 - acc: 0.9535 - val_loss: 0.4790 - val_acc: 0.8244
# Epoch 9/10
# 157/157 [==============================] - 7s 45ms/step - loss: 0.1356 - acc: 0.9549 - val_loss: 0.4374 - val_acc: 0.8746
# Epoch 10/10
# 157/157 [==============================] - 7s 45ms/step - loss: 0.1293 - acc: 0.9577 - val_loss: 0.4362 - val_acc: 0.8696

# plotting
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
# plt.show()
plt.savefig('LSTMBidirectional.png', bbox_inches='tight')