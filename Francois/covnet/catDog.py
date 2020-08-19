import os, shutil

### creating a small dataset from Original dataset

original_dataset_dir = 'D:\KaggleDataSet\dogs-vs-cats\\train'

base_dir = 'D:\KaggleDataSet\cats_and_dogs_small'
# os.mkdir(base_dir)
#
train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)
#
train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)
#
train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)
#
validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)
#
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)
#
test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)
#
test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)
# fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)

# print('total training cat images:', len(os.listdir(train_cats_dir)))
# print('total training dog images:', len(os.listdir(train_dogs_dir)))
# print('total validation cat images:', len(os.listdir(validation_cats_dir)))
# print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
# print('total test cat images:', len(os.listdir(test_cats_dir)))
# print('total test dog images:', len(os.listdir(test_dogs_dir)))
# total training cat images: 1000
# total training dog images: 1000
# total validation cat images: 500
# total validation dog images: 500
# total test cat images: 500
# total test dog images: 500

### Creating a convnet model

# from keras import layers
# from keras import models
#
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# print(model.summary())
#
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 148, 148, 32)      896
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32)        0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 72, 72, 64)        18496
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64)        0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 34, 34, 128)       73856
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 17, 17, 128)       0
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 15, 15, 128)       147584
# _________________________________________________________________
# max_pooling2d_4 (MaxPooling2 (None, 7, 7, 128)         0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 6272)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 512)               3211776
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 513
# =================================================================
# Total params: 3,453,121
# Trainable params: 3,453,121
# Non-trainable params: 0
# _________________________________________________________________
# None
#
# Process finished with exit code 0

### Setting optimizers

# from keras import optimizers
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['acc'])

### Using ImageDataGenerator to read images from directories

# from keras.preprocessing.image import ImageDataGenerator
#
# train_datagen = ImageDataGenerator(rescale=1./255)
# validation_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(150, 150),
#         batch_size=20,
#         class_mode='binary')
#
# validation_generator = validation_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=20,
#         class_mode='binary')
#
# # for data_batch, labels_batch in train_generator:
# #     print('data batch shape:', data_batch.shape)
# #     print('labels batch shape:', labels_batch.shape)
# #     break
# # Found 2000 images belonging to 2 classes.
# # Found 1000 images belonging to 2 classes.
# # data batch shape: (20, 150, 150, 3)
# # labels batch shape: (20,)
#
# ### Fitting the model using a batch generator
#
# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=100,
#       epochs=30,
#       validation_data=validation_generator,
#       validation_steps=50)
#
# model.save('cats_and_dogs_small_1.h5')
#
# ### Displaying curves of loss and accuracy during training
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

### Defining a new convnet that includes dropout

# from keras import layers
# from keras import models
# from keras import optimizers
#
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu',
#                         input_shape=(150, 150, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-4),
#               metrics=['acc'])
#
# ### Training the convnet using data-augmentation generators
#
# from keras.preprocessing.image import ImageDataGenerator
#
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,)
#
# validation_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
#
# validation_generator = validation_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
#
# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=100,
#       epochs=100,
#       validation_data=validation_generator,
#       validation_steps=50)
#
# model.save('cats_and_dogs_small_2.h5')
#
# ### Displaying curves of loss and accuracy during training
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



