import os

data_dir = 'D:\\DataSet\\jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

# Generator
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)

test_steps = (len(float_data) - 300001 - lookback)

# GRU model

# from keras.models import Sequential
# from keras import layers
# from keras.optimizers import RMSprop
#
# model = Sequential()
# model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=20,
#                               validation_data=val_gen,
#                               validation_steps=10000)
# model.save('gru.h5')
#
# # plotting
# import matplotlib.pyplot as plt
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(loss) + 1)
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.savefig('gru.png', bbox_inches='tight')
# # plt.show()

# Gru with dropout

# from tensorflow.keras.models import Sequential
# from tensorflow.keras import layers
# from tensorflow.keras.optimizers import RMSprop
#
# model = Sequential()
# model.add(layers.GRU(32,
#                      dropout=0.2,
#                      recurrent_dropout=0.2,
#                      input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen,
#                               steps_per_epoch=500,
#                               epochs=40,
#                               validation_data=val_gen,
#                               validation_steps=1000)
# model.save('gruDropout.h5')
# Epoch 1/40
# 500/500 [==============================] - 482s 965ms/step - loss: 0.3368 - val_loss: 0.2763
# Epoch 2/40
# 500/500 [==============================] - 482s 965ms/step - loss: 0.3022 - val_loss: 0.2808
# Epoch 3/40
# 500/500 [==============================] - 479s 958ms/step - loss: 0.2977 - val_loss: 0.2699
# Epoch 4/40
# 500/500 [==============================] - 483s 966ms/step - loss: 0.2928 - val_loss: 0.2709
# Epoch 5/40
# 500/500 [==============================] - 725s 1s/step - loss: 0.2891 - val_loss: 0.2759
# Epoch 6/40
# 500/500 [==============================] - 492s 985ms/step - loss: 0.2838 - val_loss: 0.2687
# Epoch 7/40
# 500/500 [==============================] - 489s 978ms/step - loss: 0.2806 - val_loss: 0.2668
# Epoch 8/40
# 500/500 [==============================] - 484s 968ms/step - loss: 0.2770 - val_loss: 0.2686
# Epoch 9/40
# 500/500 [==============================] - 488s 976ms/step - loss: 0.2728 - val_loss: 0.2682
# Epoch 10/40
# 500/500 [==============================] - 487s 975ms/step - loss: 0.2711 - val_loss: 0.2694
# Epoch 11/40
# 500/500 [==============================] - 479s 959ms/step - loss: 0.2685 - val_loss: 0.2722
# Epoch 12/40
# 500/500 [==============================] - 475s 950ms/step - loss: 0.2644 - val_loss: 0.2771
# Epoch 13/40
# 500/500 [==============================] - 475s 951ms/step - loss: 0.2624 - val_loss: 0.2824
# Epoch 14/40
# 500/500 [==============================] - 473s 945ms/step - loss: 0.2595 - val_loss: 0.2860
# Epoch 15/40
# 500/500 [==============================] - 479s 957ms/step - loss: 0.2571 - val_loss: 0.2847
# Epoch 16/40
# 500/500 [==============================] - 579s 1s/step - loss: 0.2556 - val_loss: 0.2874
# Epoch 17/40
# 500/500 [==============================] - 1020s 2s/step - loss: 0.2531 - val_loss: 0.2870
# Epoch 18/40
# 500/500 [==============================] - 986s 2s/step - loss: 0.2516 - val_loss: 0.2992
# Epoch 19/40
# 500/500 [==============================] - 517s 1s/step - loss: 0.2503 - val_loss: 0.2941
# Epoch 20/40
# 500/500 [==============================] - 490s 979ms/step - loss: 0.2469 - val_loss: 0.3014
# Epoch 21/40
# 500/500 [==============================] - 485s 969ms/step - loss: 0.2457 - val_loss: 0.3117
# Epoch 22/40
# 500/500 [==============================] - 488s 976ms/step - loss: 0.2429 - val_loss: 0.2976
# Epoch 23/40
# 500/500 [==============================] - 499s 999ms/step - loss: 0.2422 - val_loss: 0.3083
# Epoch 24/40
# 500/500 [==============================] - 479s 958ms/step - loss: 0.2413 - val_loss: 0.3098
# Epoch 25/40
# 500/500 [==============================] - 482s 965ms/step - loss: 0.2392 - val_loss: 0.3161
# Epoch 26/40
# 500/500 [==============================] - 483s 966ms/step - loss: 0.2374 - val_loss: 0.3134
# Epoch 27/40
# 500/500 [==============================] - 499s 998ms/step - loss: 0.2359 - val_loss: 0.3153
# Epoch 28/40
# 500/500 [==============================] - 723s 1s/step - loss: 0.2337 - val_loss: 0.3228
# Epoch 29/40
# 500/500 [==============================] - 681s 1s/step - loss: 0.2348 - val_loss: 0.3181
# Epoch 30/40
# 500/500 [==============================] - 675s 1s/step - loss: 0.2306 - val_loss: 0.3158
# Epoch 31/40
# 500/500 [==============================] - 762s 2s/step - loss: 0.2307 - val_loss: 0.3179
# Epoch 32/40
# 500/500 [==============================] - 940s 2s/step - loss: 0.2287 - val_loss: 0.3231
# Epoch 33/40
# 500/500 [==============================] - 671s 1s/step - loss: 0.2280 - val_loss: 0.3200
# Epoch 34/40
# 500/500 [==============================] - 477s 955ms/step - loss: 0.2281 - val_loss: 0.3217
# Epoch 35/40
# 500/500 [==============================] - 477s 954ms/step - loss: 0.2255 - val_loss: 0.3236
# Epoch 36/40
# 500/500 [==============================] - 477s 953ms/step - loss: 0.2280 - val_loss: 0.3308
# Epoch 37/40
# 500/500 [==============================] - 494s 988ms/step - loss: 0.2242 - val_loss: 0.3216
# Epoch 38/40
# 500/500 [==============================] - 484s 967ms/step - loss: 0.2225 - val_loss: 0.3298
# Epoch 39/40
# 500/500 [==============================] - 481s 961ms/step - loss: 0.2235 - val_loss: 0.3256
# Epoch 40/40
# 500/500 [==============================] - 480s 961ms/step - loss: 0.2208 - val_loss: 0.3306

# two layer GRU

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit(train_gen,
                    steps_per_epoch=500,
                    epochs=20,
                    validation_data=val_gen,
                    validation_steps=1000)
model.save('gru2layerDropout.h5')

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
plt.savefig('gru2layerDropout.png', bbox_inches='tight')
