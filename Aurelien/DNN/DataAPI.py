import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import pandas as pd

# dataset = tf.data.Dataset.range(10)
# for item in dataset:
#     print(item)
# tf.Tensor(0, shape=(), dtype=int64)
# tf.Tensor(1, shape=(), dtype=int64)
# tf.Tensor(2, shape=(), dtype=int64)
# tf.Tensor(3, shape=(), dtype=int64)
# tf.Tensor(4, shape=(), dtype=int64)
# tf.Tensor(5, shape=(), dtype=int64)
# tf.Tensor(6, shape=(), dtype=int64)
# tf.Tensor(7, shape=(), dtype=int64)
# tf.Tensor(8, shape=(), dtype=int64)
# tf.Tensor(9, shape=(), dtype=int64)

# dataset = dataset.repeat(3).batch(7)
# for item in dataset:
#     print(item)
# tf.Tensor([0 1 2 3 4 5 6], shape=(7,), dtype=int64)
# tf.Tensor([7 8 9 0 1 2 3], shape=(7,), dtype=int64)
# tf.Tensor([4 5 6 7 8 9 0], shape=(7,), dtype=int64)
# tf.Tensor([1 2 3 4 5 6 7], shape=(7,), dtype=int64)
# tf.Tensor([8 9], shape=(2,), dtype=int64)

# dataset = dataset.map(lambda x: x * 2)
# for item in dataset:
#     print(item)
# tf.Tensor([ 0  2  4  6  8 10 12], shape=(7,), dtype=int64)
# tf.Tensor([14 16 18  0  2  4  6], shape=(7,), dtype=int64)
# tf.Tensor([ 8 10 12 14 16 18  0], shape=(7,), dtype=int64)
# tf.Tensor([ 2  4  6  8 10 12 14], shape=(7,), dtype=int64)
# tf.Tensor([16 18], shape=(2,), dtype=int64)

# dataset = dataset.unbatch()
# dataset = dataset.filter(lambda x: x < 10)
# for item in dataset.take(3):
#     print(item)
# tf.Tensor(0, shape=(), dtype=int64)
# tf.Tensor(2, shape=(), dtype=int64)
# tf.Tensor(4, shape=(), dtype=int64

# Shuffling the Data

# tf.random.set_seed(42)
# dataset = tf.data.Dataset.range(10).repeat(3)
# dataset = dataset.shuffle(buffer_size=3, seed=42).batch(7)
# for item in dataset:
#     print(item)

### Split the dataset to multiple CSV files

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target.reshape(-1, 1),
                                                              random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_mean = scaler.mean_
X_std = scaler.scale_

# def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
#     housing_dir = os.path.join("datasets", "housing")
#     os.makedirs(housing_dir, exist_ok=True)
#     path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")
#
#     filepaths = []
#     m = len(data)
#     for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
#         part_csv = path_format.format(name_prefix, file_idx)
#         filepaths.append(part_csv)
#         with open(part_csv, "wt", encoding="utf-8") as f:
#             if header is not None:
#                 f.write(header)
#                 f.write("\n")
#             for row_idx in row_indices:
#                 f.write(",".join([repr(col) for col in data[row_idx]]))
#                 f.write("\n")
#     return filepaths
# train_data = np.c_[X_train, y_train]
# valid_data = np.c_[X_valid, y_valid]
# test_data = np.c_[X_test, y_test]
# header_cols = housing.feature_names + ["MedianHouseValue"]
# header = ",".join(header_cols)
#
# train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
# valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
# test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)
# print(train_filepaths)
# print(valid_filepaths)
# print(test_filepaths)
path_format = os.path.join(os.path.join("datasets","housing"), "my_{}_{:02d}.csv")
train_filepaths = []
for i in range(20):
    train_filepaths.append(path_format.format("train", i))
valid_filepaths = []
test_filepaths = []
for i in range(10):
    valid_filepaths.append(path_format.format("valid", i))
    test_filepaths.append(path_format.format("test", i))

# print(pd.read_csv(train_filepaths[0]).head())
#    MedInc  HouseAge  AveRooms  ...  Latitude  Longitude  MedianHouseValue
# 0  3.5214      15.0  3.049945  ...     37.63    -122.43             1.442
# 1  5.3275       5.0  6.490060  ...     33.69    -117.39             1.687
# 2  3.1000      29.0  7.542373  ...     38.44    -122.98             1.621
# 3  7.1736      12.0  6.289003  ...     33.55    -117.70             2.621
# 4  2.0549      13.0  5.312457  ...     33.93    -116.93             0.956

# with open(train_filepaths[0]) as f:
#     for i in range(5):
#         print(f.readline(), end="")
# [5 rows x 9 columns]
# MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,MedianHouseValue
# 3.5214,15.0,3.0499445061043287,1.106548279689234,1447.0,1.6059933407325193,37.63,-122.43,1.442
# 5.3275,5.0,6.490059642147117,0.9910536779324056,3464.0,3.4433399602385686,33.69,-117.39,1.687
# 3.1,29.0,7.5423728813559325,1.5915254237288134,1328.0,2.2508474576271187,38.44,-122.98,1.621
# 7.1736,12.0,6.289002557544757,0.9974424552429667,1054.0,2.6956521739130435,33.55,-117.7,2.621

# Shuffling

# filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42) # list_files shuffles the file path
# for filepath in filepath_dataset:
#     print(filepath)
# tf.Tensor(b'datasets\\housing\\my_train_05.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_16.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_01.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_17.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_00.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_14.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_10.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_02.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_12.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_19.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_07.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_09.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_13.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_15.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_11.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_18.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_04.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_06.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_03.csv', shape=(), dtype=string)
# tf.Tensor(b'datasets\\housing\\my_train_08.csv', shape=(), dtype=string)

# n_readers = 5
# dataset = filepath_dataset.interleave(lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
#                                       cycle_length=n_readers)
# for line in dataset.take(5):
#     print(line.numpy())
# b'4.5909,16.0,5.475877192982456,1.0964912280701755,1357.0,2.9758771929824563,33.63,-117.71,2.418'
# b'2.4792,24.0,3.4547038327526134,1.1341463414634145,2251.0,3.921602787456446,34.18,-118.38,2.0'
# b'4.2708,45.0,5.121387283236994,0.953757225433526,492.0,2.8439306358381504,37.48,-122.19,2.67'
# b'2.1856,41.0,3.7189873417721517,1.0658227848101265,803.0,2.0329113924050635,32.76,-117.12,1.205'
# b'4.1812,52.0,5.701388888888889,0.9965277777777778,692.0,2.4027777777777777,33.73,-118.31,3.215'

# setting defaults

# record_defaults=[0, np.nan, tf.constant(np.nan, dtype=tf.float64), "Hello", tf.constant([])]
# parsed_fields = tf.io.decode_csv('1,2,3,4,5', record_defaults)
# print(parsed_fields)
# [<tf.Tensor: shape=(), dtype=int32, numpy=1>,
# <tf.Tensor: shape=(), dtype=float32, numpy=2.0>,
# <tf.Tensor: shape=(), dtype=float64, numpy=3.0>,
# <tf.Tensor: shape=(), dtype=string, numpy=b'4'>,
# <tf.Tensor: shape=(), dtype=float32, numpy=5.0>]
# parsed_fields = tf.io.decode_csv(',,,,5', record_defaults)
# print(parsed_fields)
# [<tf.Tensor: shape=(), dtype=int32, numpy=0>,
# <tf.Tensor: shape=(), dtype=float32, numpy=nan>,
# <tf.Tensor: shape=(), dtype=float64, numpy=nan>,
# <tf.Tensor: shape=(), dtype=string, numpy=b'Hello'>,
# <tf.Tensor: shape=(), dtype=float32, numpy=5.0>]

# preprocessing

n_inputs = 8 # X_train.shape[-1]
@tf.function
def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y

# print(preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782'))
# (<tf.Tensor: shape=(8,), dtype=float32, numpy=
# array([ 0.16579157,  1.216324  , -0.05204565, -0.39215982, -0.5277444 ,
#        -0.2633488 ,  0.8543046 , -1.3072058 ], dtype=float32)>,
# <tf.Tensor: shape=(1,), dtype=float32, numpy=array([2.782], dtype=float32)>)

#  data read

def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

# tf.random.set_seed(42)
# train_set = csv_reader_dataset(train_filepaths, batch_size=3)
# for X_batch, y_batch in train_set.take(2):
#     print("X =", X_batch)
#     print("y =", y_batch)
#     print()
# X = tf.Tensor(
# [[ 0.5804519  -0.20762321  0.05616303 -0.15191229  0.01343246  0.00604472
#    1.2525111  -1.3671792 ]
#  [ 5.818099    1.8491895   1.1784915   0.28173092 -1.2496178  -0.3571987
#    0.7231292  -1.0023477 ]
#  [-0.9253566   0.5834586  -0.7807257  -0.28213993 -0.36530012  0.27389365
#   -0.76194876  0.72684526]], shape=(3, 8), dtype=float32)
# y = tf.Tensor(
# [[1.752]
#  [1.313]
#  [1.535]], shape=(3, 1), dtype=float32)
#
# X = tf.Tensor(
# [[-0.8324941   0.6625668  -0.20741376 -0.18699841 -0.14536144  0.09635526
#    0.9807942  -0.67250353]
#  [-0.62183803  0.5834586  -0.19862501 -0.3500319  -1.1437552  -0.3363751
#    1.107282   -0.8674123 ]
#  [ 0.8683102   0.02970133  0.3427381  -0.29872298  0.7124906   0.28026953
#   -0.72915536  0.86178064]], shape=(3, 8), dtype=float32)
# y = tf.Tensor(
# [[0.919]
#  [1.028]
#  [2.182]], shape=(3, 1), dtype=float32)

# using tf.keras

train_set = csv_reader_dataset(train_filepaths, repeat=None)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
                                 keras.layers.Dense(1)])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
batch_size = 32
model.fit(train_set, steps_per_epoch=len(X_train) // batch_size, epochs=10, validation_data=valid_set)
# loss: 0.4826 - val_loss: 0.4714
model.evaluate(test_set, steps=len(X_test) // batch_size)
# loss: 0.4788

new_set = test_set.map(lambda X, y: X) # we could instead just pass test_set, Keras would ignore the labels
X_new = X_test
print(model.predict(new_set, steps=len(X_new) // batch_size))
# [[3.837813 ]
#  [2.395659 ]
#  [1.4261606]
#  ...
#  [1.6820569]
#  [1.8740587]
#  [0.765728 ]]


