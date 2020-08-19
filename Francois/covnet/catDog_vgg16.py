
### Instantiating the VGG16 convolutional base

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
# print(conv_base.summary())
#
# Model: "vgg16"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_1 (InputLayer)         (None, 150, 150, 3)       0
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 150, 150, 64)      1792
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 150, 150, 64)      36928
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 75, 75, 64)        0
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 75, 75, 128)       73856
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 75, 75, 128)       147584
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 37, 37, 128)       0
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 37, 37, 256)       295168
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 37, 37, 256)       590080
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 37, 37, 256)       590080
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 18, 18, 256)       0
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 18, 18, 512)       1180160
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 18, 18, 512)       2359808
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 18, 18, 512)       2359808
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 9, 9, 512)         0
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 9, 9, 512)         2359808
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 9, 9, 512)         2359808
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 9, 9, 512)         2359808
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 4, 4, 512)         0
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________
# None
#
# Process finished with exit code 0

### Extracting features using the pretrained convolutional base

# import os
# import numpy as np
# from keras.preprocessing.image import ImageDataGenerator
#
# base_dir = 'D:\KaggleDataSet\cats_and_dogs_small'
# train_dir = os.path.join(base_dir, 'train')
# validation_dir = os.path.join(base_dir, 'validation')
# test_dir = os.path.join(base_dir, 'test')
#
# datagen = ImageDataGenerator(rescale=1./255)
# batch_size = 20
#
# def extract_features(directory, sample_count):
#     features = np.zeros(shape=(sample_count, 4, 4, 512))
#     labels = np.zeros(shape=(sample_count))
#     generator = datagen.flow_from_directory(
#         directory,
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')
#     i = 0
#     for inputs_batch, labels_batch in generator:
#         features_batch = conv_base.predict(inputs_batch)
#         features[i * batch_size : (i + 1) * batch_size] = features_batch
#         labels[i * batch_size : (i + 1) * batch_size] = labels_batch
#         i += 1
#         if i * batch_size >= sample_count:
#             break
#     return features, labels
#
# train_features, train_labels = extract_features(train_dir, 2000)
# validation_features, validation_labels = extract_features(validation_dir, 1000)
# test_features, test_labels = extract_features(test_dir, 1000)
#
# train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
# validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
# test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
#
# ### Defining and training the densely connected classifier
#
# from keras import models
# from keras import layers
# from keras import optimizers
#
# model = models.Sequential()
# model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
#               loss='binary_crossentropy',
#               metrics=['acc'])
#
# history = model.fit(train_features, train_labels,
#                     epochs=30,
#                     batch_size=20,
#                     validation_data=(validation_features, validation_labels))
#
# ### Plotting
#
# import matplotlib.pyplot as plt
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()

### Adding a densely connected classifier on top of the convolutional base

from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# print(model.summary())
#
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# vgg16 (Model)                (None, 4, 4, 512)         14714688
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 8192)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 256)               2097408
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 257
# =================================================================
# Total params: 16,812,353
# Trainable params: 16,812,353
# Non-trainable params: 0
# _________________________________________________________________
# None

### Training the model end to end with a frozen convolutional base

import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

base_dir = 'D:\KaggleDataSet\cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50)
# loss: 0.0320 - acc: 0.9915 - val_loss: 0.4240 - val_acc: 0.9660

# ### Plotting

# import matplotlib.pyplot as plt
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()

### Fine-tuning the model

### Freezing all layers up to a specific one
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Fine-tuning the model

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)
# loss: 0.0029 - acc: 0.9990 - val_loss: 0.0873 - val_acc: 0.9700

model.save('cats_and_dogs_small_vgg16.h5')

### Plotting

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

plt.show()

### Running the model on test dataset

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
# test acc: 0.9660000205039978