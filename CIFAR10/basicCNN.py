import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

data_dirpath = "C:/Users/Ace/.keras/datasets/"

def unpickle(file):
    import pickle
    file = os.path.join(data_dirpath, "cifar-10-batches-py/", file)
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(filename):

    data = unpickle(filename)

    batch_images = data[b'data']
    batch_images = np.array(batch_images, dtype=float) / 255.0
    batch_images = batch_images.reshape([-1, 3, 32, 32])
    batch_images = batch_images.transpose([0, 2, 3, 1])

    batch_labels = np.array(data[b'labels'])

    return batch_images, batch_labels

def load_training_data():

    training_images = np.zeros(shape=[50000, 32, 32, 3], dtype=float)
    training_labels = np.zeros(shape=[50000], dtype=int)
    start_index = 0

    for i in range(5):
        batch_images, batch_labels = load_data(filename="data_batch_" + str(i + 1))
        end_index = start_index + len(batch_images)

        training_images[start_index:end_index, :] = batch_images

        training_labels[start_index:end_index] = batch_labels

        start_index = end_index

    return training_images, training_labels

def load_test_data():

    batch_images, batch_labels = load_data(filename="test_batch")
    return batch_images, batch_labels

x_train, y_train = load_training_data()
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

x_test, y_test = load_test_data()
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

batch_size = 32
num_classes = 10
epochs = 5
num_predictions = 20

# file = "C:\\Users\\Ace\\.keras\\datasets\\cifar-10-batches-py\\data_batch_1"
# x = unpickle(file)
# print(x.keys())
# y = np.array(x[b'data'])
# print(type(y))
# print(y.shape)
# print(y[0])
# print("Min: ",np.amin(y[0])," Max: ",np.amax(y[0]))
# y = np.array(y, dtype=float) / 255.0
# print("Min: ",np.amin(y[0])," Max: ",np.amax(y[0]))
# y = y.reshape([-1, 3, 32, 32])
# print((y.shape))
# y = y.transpose([0, 2, 3, 1])
# print((y.shape))
# The data, split between train and test sets:
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.load_weights('cnn15.h5')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
model.save_weights('cnn20.h5')

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])