import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string], '')
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

encoder = info.features['text'].encoder
print ('Vocabulary size: {}'.format(encoder.vocab_size))

# sample_string = 'Just do it!'
# encoded_string = encoder.encode(sample_string)
# print ('Encoded string is {}'.format(encoded_string))
# original_string = encoder.decode(encoded_string)
# print ('The original string: "{}"'.format(original_string))
# for index in encoded_string:
#   print ('{} ----> {}'.format(index, encoder.decode([index])))

BUFFER_SIZE = 10000
BATCH_SIZE = 64
EVALUATION_INTERVAL = 50

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, train_dataset.output_shapes)
test_dataset = test_dataset.padded_batch(BATCH_SIZE, test_dataset.output_shapes)

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(encoder.vocab_size, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
# print(model.summary())
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# history = model.fit(train_dataset, epochs=5, steps_per_epoch=EVALUATION_INTERVAL)
# model.save('rnn_model1.h5')

model = tf.keras.models.load_model('rnn_model1.h5')

# test_loss, test_acc = model.evaluate(test_dataset)
#
# print('Test Loss: {}'.format(test_loss))
# print('Test Accuracy: {}'.format(test_acc))

def pad_to_size(vec, size):
  zeros = [0] * (size - len(vec))
  vec.extend(zeros)
  return vec

def sample_predict(sentence, pad):
  encoded_sample_pred_text = encoder.encode(sentence)

  if pad:
    encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
  encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
  predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))

  return (predictions)

sample_pred_text = ('The movie was bad. The animation and the graphics were rubbish. I would never recommend this movie.')
predictions = sample_predict(sample_pred_text, pad=True)
print (predictions)

