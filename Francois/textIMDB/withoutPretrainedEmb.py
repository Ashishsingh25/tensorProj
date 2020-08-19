# Processing the labels of the raw IMDB data

import os

imdb_dir = 'D:\\KaggleDataSet\\aclImdb\\aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding="utf8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# Tokenizing the text of the raw IMDB data

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100
# training_samples = 200
training_samples = 20000
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
# Found 88582 unique tokens.

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)
# Shape of data tensor: (25000, 100)
# Shape of label tensor: (25000,)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

# Training the model without pretrained word embeddings
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

embedding_dim = 100

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.summary()
# Model: "sequential_1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 100, 100)          1000000
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 10000)             0
# _________________________________________________________________
# dense_1 (Dense)              (None, 32)                320032
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 33
# =================================================================
# Total params: 1,320,065
# Trainable params: 1,320,065
# Non-trainable params: 0
# _________________________________________________________________

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
# with 200 training samples
# loss: 0.0029 - acc: 1.0000 - val_loss: 0.7572 - val_acc: 0.5211

# with 20000 training samples
# loss: 1.9586e-09 - acc: 1.0000 - val_loss: 1.2849 - val_acc: 0.8248

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

plt.show()