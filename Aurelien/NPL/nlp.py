import tensorflow as tf
import tensorflow_datasets as tfds
from collections import Counter
from tensorflow import keras
import numpy as np


# shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
# with open(filepath) as f:
#     shakespeare_text = f.read()
# print(shakespeare_text[:148])
# First Citizen:
# Before we proceed any further, hear me speak.
#
# All:
# Speak, speak.
#
# First Citizen:
# You are all resolved rather to die than to famish?
# print("".join(sorted(set(shakespeare_text.lower())))) # uniq characters
# !$ & ',-.3:;?abcdefghijklmnopqrstuvwxyz

# tokenize

# tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
# tokenizer.fit_on_texts(shakespeare_text)
# # print(tokenizer.texts_to_sequences(["First"]))
# # [[20, 6, 9, 8, 3]]
# # print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))
# # ['f i r s t']
# max_id = len(tokenizer.word_index) # number of distinct characters
# dataset_size = tokenizer.document_count # total number of characters
# # print(max_id)
# # 39
# # print(dataset_size)
# # 1115394

# creating train, val and test sets

# [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
# # print(encoded[:10])
# train_size = dataset_size * 90 // 100
# dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
# # for element in dataset.take(10):
# #   print(element)
# # print("_________________")
# n_steps = 100
# window_length = n_steps + 1 # target = input shifted 1 character ahead
# dataset = dataset.repeat().window(window_length, shift=1, drop_remainder=True)
# # for element in dataset.take(2):
# #     print("++++++++++++++++++++++++++++++++")
# #     for ele in element:
# #         print(ele)
# # flat_map -> nested dataset into a flat dataset
# dataset = dataset.flat_map(lambda window: window.batch(window_length))
# # for element in dataset.take(2):
# #   print(element)
# np.random.seed(42)
# tf.random.set_seed(42)
# batch_size = 32
# dataset = dataset.shuffle(10000).batch(batch_size)
# dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
# # for element in dataset.take(1):
# #   print(element)
# dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
# dataset = dataset.prefetch(1)
# # for X_batch, Y_batch in dataset.take(1):
# #     print(X_batch.shape, Y_batch.shape)
# # (32, 100, 39) (32, 100)
# # for X_batch, Y_batch in dataset.take(1):
# #     print(X_batch[1,:3,:], Y_batch[0,:3])

# model = keras.models.Sequential([
#     keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id], dropout=0.2, recurrent_dropout=0.2),
#     keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
#     keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))])
# model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
# history = model.fit(dataset, steps_per_epoch=train_size // batch_size, epochs=10)
# model.save("my_shakespeare_model.h5")

# Generate Text

# def preprocess(texts):
#     X = np.array(tokenizer.texts_to_sequences(texts)) - 1
#     return tf.one_hot(X, max_id)

# X_new = preprocess(["How are yo"])
# Y_pred = model.predict_classes(X_new)
# print(tokenizer.sequences_to_texts(Y_pred + 1)[0][-1]) # 1st sentence, last char

# def next_char(text, temperature=1):
#     X_new = preprocess([text])
#     y_proba = model.predict(X_new)[0, -1:, :]
#     rescaled_logits = tf.math.log(y_proba) / temperature
#     char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
#     return tokenizer.sequences_to_texts(char_id.numpy())[0]
# def complete_text(text, n_chars=50, temperature=1):
#     for _ in range(n_chars):
#         text += next_char(text, temperature)
#     return text

# print(complete_text("t", temperature=0.2))
# print(complete_text("t", temperature=1))
# print(complete_text("t", temperature=2))

### stateful RNN

# batch_size = 32
# encoded_parts = np.array_split(encoded[:train_size], batch_size)
# datasets = []
# for encoded_part in encoded_parts:
#     dataset = tf.data.Dataset.from_tensor_slices(encoded_part)
#     dataset = dataset.window(window_length, shift=n_steps, drop_remainder=True)
#     dataset = dataset.flat_map(lambda window: window.batch(window_length))
#     datasets.append(dataset)
# dataset = tf.data.Dataset.zip(tuple(datasets)).map(lambda *windows: tf.stack(windows))
# dataset = dataset.repeat().map(lambda windows: (windows[:, :-1], windows[:, 1:]))
# dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
# dataset = dataset.prefetch(1)
#
# model = keras.models.Sequential([
#     keras.layers.GRU(128, return_sequences=True, stateful=True,
#                      dropout=0.2, recurrent_dropout=0.2,
#                      batch_input_shape=[batch_size, None, max_id]),
#     keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2, recurrent_dropout=0.2),
#     keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))])
# class ResetStatesCallback(keras.callbacks.Callback):
#     def on_epoch_begin(self, epoch, logs):
#         self.model.reset_states()
# model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
# steps_per_epoch = train_size // batch_size // n_steps
# history = model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=10, callbacks=[ResetStatesCallback()])
# # loss: 1.7583
# model.save("my_shakespeare_stateless_model.h5")
# # to make prediction, we need stateless copy
#
# stateless_model = keras.models.Sequential([
#     keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id]),
#     keras.layers.GRU(128, return_sequences=True),
#     keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))])
# stateless_model.build(tf.TensorShape([None, None, max_id]))
# stateless_model.set_weights(model.get_weights())
# model = stateless_model
# tf.random.set_seed(42)
#
# print(complete_text("t", temperature=0.2))
# # ther a do my lords
# # and the confort the soul of the
# print(complete_text("t", temperature=1))
# # to3ccops
# # do take throble constrant some to the shal
# print(complete_text("t", temperature=2))
# # tpend mevelv!
# # wel? my deself.'d getderin'rs:
# # -haper

### Sentiment Analysis

# tf.random.set_seed(42)
#
# datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)
# print(datasets.keys())
# # dict_keys(['test', 'train', 'unsupervised'])
#
# train_size = info.splits["train"].num_examples
# test_size = info.splits["test"].num_examples
#
# for X_batch, y_batch in datasets["train"].batch(2).take(1):
#     for review, label in zip(X_batch.numpy(), y_batch.numpy()):
#         print("Review:", review.decode("utf-8")[:200], "...")
#         print("Label:", label, "= Positive" if label else "= Negative")
#         print()
# # Review: This is a big step down after the surprisingly enjoyable original. This sequel isn't nearly as fun as part
# # one, and it instead spends too much time on plot development. Tim Thomerson is still the best ...
# # Label: 0 = Negative
# #
# # Review: Perhaps because I was so young, innocent and BRAINWASHED when I saw it, this movie was the cause of many
# # sleepless nights for me. I haven't seen it since I was in seventh grade at a Presbyterian schoo ...
# # Label: 0 = Negative
#
# def preprocess(X_batch, y_batch):
#     X_batch = tf.strings.substr(X_batch, 0, 300)
#     X_batch = tf.strings.regex_replace(X_batch, rb"<br\s*/?>", b" ")
#     X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
#     X_batch = tf.strings.split(X_batch)
#     return X_batch.to_tensor(default_value=b"<pad>"), y_batch
#
# print(preprocess(X_batch, y_batch))
# # (<tf.Tensor: shape=(2, 60), dtype=string, numpy=
# # array([[b'This', b'is', b'a', b'big', b'step', b'down', b'after', b'the',
# #         b'surprisingly', b'enjoyable', b'original', b'This', b'sequel',
# #         b"isn't", b'nearly', b'as', b'fun', b'as', b'part', b'one',
# #         b'and', b'it', b'instead', b'spends', b'too', b'much', b'time',
# #         b'on', b'plot', b'development', b'Tim', b'Thomerson', b'is',
# #         b'still', b'the', b'best', b'thing', b'about', b'this',
# #         b'series', b'but', b'his', b'wisecracking', b'is', b'toned',
# #         b'down', b'in', b'this', b'entry', b'The', b'performances',
# #         b'are', b'all', b'<pad>', b'<pad>', b'<pad>', b'<pad>', b'<pad>',
# #         b'<pad>', b'<pad>'],
# #        [b'Perhaps', b'because', b'I', b'was', b'so', b'young',
# #         b'innocent', b'and', b'BRAINWASHED', b'when', b'I', b'saw',
# #         b'it', b'this', b'movie', b'was', b'the', b'cause', b'of',
# #         b'many', b'sleepless', b'nights', b'for', b'me', b'I',
# #         b"haven't", b'seen', b'it', b'since', b'I', b'was', b'in',
# #         b'seventh', b'grade', b'at', b'a', b'Presbyterian', b'school',
# #         b'so', b'I', b'am', b'not', b'sure', b'what', b'effect', b'it',
# #         b'would', b'have', b'on', b'me', b'now', b'However', b'I',
# #         b'will', b'say', b'that', b'it', b'left', b'an', b'impress']],
# #       dtype=object)>, <tf.Tensor: shape=(2,), dtype=int64, numpy=array([0, 0], dtype=int64)>)
#
# vocabulary = Counter()
# for X_batch, y_batch in datasets["train"].batch(32).map(preprocess):
#     for review in X_batch:
#         vocabulary.update(list(review.numpy()))
#
# print(vocabulary.most_common()[:3])
# # [(b'<pad>', 214741), (b'the', 61137), (b'a', 38564)]
# print(len(vocabulary))
# # 53893
#
# vocab_size = 10000
# truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]
# word_to_id = {word: index for index, word in enumerate(truncated_vocabulary)}
# for word in b"This movie was faaaaaantastic".split():
#     print(word_to_id.get(word) or vocab_size)
# # 22
# # 12
# # 11
# # 10000
#
# words = tf.constant(truncated_vocabulary)
# word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
# vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
# num_oov_buckets = 1000
# table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)
# print(table.lookup(tf.constant([b"This movie was faaaaaantastic".split()])))
# # tf.Tensor([[   22    12    11 10053]], shape=(1, 4), dtype=int64)
#
# def encode_words(X_batch, y_batch):
#     return table.lookup(X_batch), y_batch
#
# train_set = datasets["train"].repeat().batch(32).map(preprocess)
# train_set = train_set.map(encode_words).prefetch(1)
# for X_batch, y_batch in train_set.take(1):
#     print(X_batch)
#     print(y_batch)
# # tf.Tensor(
# # [[   22     7     2 ...     0     0     0]
# #  [ 1239    82     6 ...   418    28  4245]
# #  [ 4246     3     1 ...     0     0     0]
# #  ...
# #  [   22     7    23 ...     0     0     0]
# #  [ 1297  3744     7 ...     0     0     0]
# #  [10928 10687  4537 ...     0     0     0]], shape=(32, 60), dtype=int64)
# # tf.Tensor([0 0 1 0 1 0 1 1 1 0 0 0 0 0 0 0 1 0 0 1 0 0 1 1 1 0 1 0 0 1 1 1], shape=(32,), dtype=int64)
#
# embed_size = 128
# model = keras.models.Sequential([
#     keras.layers.Embedding(vocab_size + num_oov_buckets, embed_size,
#                            mask_zero=True, # not shown in the book
#                            input_shape=[None]),
#     keras.layers.GRU(128, return_sequences=True),
#     keras.layers.GRU(128),
#     keras.layers.Dense(1, activation="sigmoid")])
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# history = model.fit(train_set, steps_per_epoch=train_size // 32, epochs=5)
# # loss: 0.0996 - accuracy: 0.9656

# Bidirectional Recurrent Layers

model = keras.models.Sequential([
    keras.layers.GRU(10, return_sequences=True, input_shape=[None, 10]),
    keras.layers.Bidirectional(keras.layers.GRU(10, return_sequences=True))])

print(model.summary())
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# gru (GRU)                    (None, None, 10)          660
# _________________________________________________________________
# bidirectional (Bidirectional (None, None, 20)          1320
# =================================================================
# Total params: 1,980
# Trainable params: 1,980
# Non-trainable params: 0

