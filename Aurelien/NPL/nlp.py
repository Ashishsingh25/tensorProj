import tensorflow as tf
from tensorflow import keras
import numpy as np


shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()
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

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)
# print(tokenizer.texts_to_sequences(["First"]))
# [[20, 6, 9, 8, 3]]
# print(tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]]))
# ['f i r s t']
max_id = len(tokenizer.word_index) # number of distinct characters
dataset_size = tokenizer.document_count # total number of characters
# print(max_id)
# 39
# print(dataset_size)
# 1115394

# creating train, val and test sets

[encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
# print(encoded[:10])
train_size = dataset_size * 90 // 100
dataset = tf.data.Dataset.from_tensor_slices(encoded[:train_size])
# for element in dataset.take(10):
#   print(element)
# print("_________________")
n_steps = 100
window_length = n_steps + 1 # target = input shifted 1 character ahead
dataset = dataset.repeat().window(window_length, shift=1, drop_remainder=True)
# for element in dataset.take(2):
#     print("++++++++++++++++++++++++++++++++")
#     for ele in element:
#         print(ele)
# flat_map -> nested dataset into a flat dataset
dataset = dataset.flat_map(lambda window: window.batch(window_length))
# for element in dataset.take(2):
#   print(element)
np.random.seed(42)
tf.random.set_seed(42)
batch_size = 32
dataset = dataset.shuffle(10000).batch(batch_size)
dataset = dataset.map(lambda windows: (windows[:, :-1], windows[:, 1:]))
# for element in dataset.take(1):
#   print(element)
dataset = dataset.map(lambda X_batch, Y_batch: (tf.one_hot(X_batch, depth=max_id), Y_batch))
dataset = dataset.prefetch(1)
# for X_batch, Y_batch in dataset.take(1):
#     print(X_batch.shape, Y_batch.shape)
# (32, 100, 39) (32, 100)
# for X_batch, Y_batch in dataset.take(1):
#     print(X_batch[1,:3,:], Y_batch[0,:3])

model = keras.models.Sequential([
    keras.layers.GRU(128, return_sequences=True, input_shape=[None, max_id], dropout=0.2, recurrent_dropout=0.2),
    keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
history = model.fit(dataset, steps_per_epoch=train_size // batch_size, epochs=10)
model.save("my_shakespeare_model.h5")

# Generate Text

def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)

X_new = preprocess(["How are yo"])
Y_pred = model.predict_classes(X_new)
print(tokenizer.sequences_to_texts(Y_pred + 1)[0][-1]) # 1st sentence, last char

def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]
def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text

print(complete_text("t", temperature=0.2))
print(complete_text("t", temperature=1))
print(complete_text("t", temperature=2))


