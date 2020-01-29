#import the packages / dependecies
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential #ANN architecture
from keras.layers import Dense # The layers in the ANN
from keras.utils import to_categorical
import matplotlib.pyplot as plt # Graph

#Load the data set
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# train_images = mnist.train_images() # training data of images
# train_labels = mnist.train_labels() # training data of the labels
# test_images = mnist. test_images()  # testing data images
# test_labels = mnist.test_labels()   # testing data labels
# print(type(train_images))
# print(train_images[0])
# plt.imshow(train_images[1], cmap='gray')
# plt.show()
# print(train_labels[0:5])

#Normalize the pixel values from [0, 255] to [-0.5 to 0.5]

# print("Min: ",np.amin(train_images[0])," Max: ",np.amax(train_images[0]))
train_images = (train_images / 255) - 0.5
test_images = (test_images/ 255) - 0.5
# print("Min: ",np.amin(train_images[0])," Max: ",np.amax(train_images[0]))
# plt.imshow(train_images[1], cmap='gray')
# plt.show()

#Flatten the images.

train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1,784))
# print(train_images.shape) #60,000 rows and 784 cols
# print(test_images.shape)  #10,000 rows and 784 cols


#Build the ANN model with 3 layers
# 2 with 64 neurons and activation function = relu
# 1 layer with 10 neurons with activation function= softmax
# model = Sequential()
# model.add(Dense(64, activation='relu', input_dim=784))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(10, activation='softmax'))

#Compile the model
# loss measures how well the model did on training, and then tries to improve on
# it using the optimizer
# model.compile(
#   optimizer= 'adam',
#     loss = 'categorical_crossentropy', #loss function for classes > 2
#     metrics = ['accuracy']
# )

#Train the model
# model.fit(
#     train_images, #The training data images
#     to_categorical(train_labels),#The trainind data labels, label data only returns a single digit representing the class of each label Ex: train_labels = 2,to_categorical(2)= [0,0,1,0,0,0,0,0,0,0]
#     epochs=5, #Number of iterations over the entire data set to train on
#     batch_size = 3 #The number of samples per gradient update for training
# )

#Evaluate the model


#save the model to disk
# model.save_weights('model.h5')
# Load the model from disk later using:
# Build the model.
model = Sequential([
 Dense(64, activation='relu', input_shape=(784,)),
 Dense(64, activation='relu'),
 Dense(10, activation='softmax'),
])
model.compile(
  optimizer= 'adam',
    loss = 'categorical_crossentropy', #loss function for classes > 2
    metrics = ['accuracy']
)
model.load_weights('model.h5')
model.summary()

print(model.evaluate(
  test_images,
  to_categorical(test_labels)
))

# Confusion Matrix
pred_labels=model.predict_classes(test_images)
conf_mat = tf.math.confusion_matrix(labels=test_labels, predictions=pred_labels)
with tf.Session():
   print('Confusion Matrix: \n\n', tf.Tensor.eval(conf_mat,feed_dict=None, session=None))

#Make predictions
# predictions = model.predict(test_images[:5])
# print(predictions)
# print (np.argmax(predictions, axis =1))
# print(test_labels[:5])

# import matplotlib.pyplot as plt
# for i in range(0,5):
#   first_image = test_images[i]
#   first_image = np.array(first_image, dtype='float')
#   pixels = first_image.reshape((28, 28))
#   plt.imshow(pixels, cmap='gray')
#   plt.show()
