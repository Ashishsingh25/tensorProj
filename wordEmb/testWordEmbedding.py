# from __future__ import absolute_import, division, print_function, unicode_literals
import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)
# print(tf.executing_eagerly())

# embedding_layer = layers.Embedding(1000, 5)
# result = embedding_layer(tf.constant([1,2,3]))
# print(type(result))

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)

# for temp in train_data.take(2):
#     print(type(temp))
#     print(temp)
# print(type(train_data))
# print((train_data))

encoder = info.features['text'].encoder
# print(encoder.subwords[:20])

padded_shapes = ([None],())
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)
# print(type(train_batches))
# for temp in train_batches.take(1):
#     print(type(temp))
#     print(temp)
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)
train_batch, train_labels = next(iter(train_batches))
# print(train_batch.numpy())

embedding_dim=16

model = keras.Sequential([
  layers.Embedding(encoder.vocab_size, embedding_dim),
  layers.GlobalAveragePooling1D(),
  layers.Dense(16, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])
#
# # model.summary()
#
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)
#
# model.save('test_model2.h5')
#
# # model = tf.keras.models.load_model('test_model1.h5')
# # model.summary()
#
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)
#
# encoder = info.features['text'].encoder
#
# out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
# out_m = io.open('meta.tsv', 'w', encoding='utf-8')
#
# for num, word in enumerate(encoder.subwords):
#   vec = weights[num+1] # skip 0, it's padding.
#   out_m.write(word + "\n")
#   out_v.write('\t'.join([str(x) for x in vec]) + "\n")
# out_v.close()
# out_m.close()
