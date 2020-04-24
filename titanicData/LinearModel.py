import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

# print(dftrain.loc[0],y_train.loc[0])
# print(dftrain.head())
# print(dftrain.describe())
# print(dftrain.shape[0], dfeval.shape[0])
# dftrain.age.hist(bins=20)

# Histogram based on age
# plt.hist(dftrain["age"],bins=20)
# plt.show()

# Bar chart based on gender
# plt.bar(np.arange(len(dftrain.sex.value_counts())), dftrain.sex.value_counts())
# plt.xticks(np.arange(len(dftrain.sex.value_counts())), dftrain.sex.value_counts().index)
# plt.show()

# Bar chart based on class
# plt.bar(np.arange(len(dftrain["class"].value_counts())), dftrain["class"].value_counts())
# plt.xticks(np.arange(len(dftrain["class"].value_counts())), dftrain["class"].value_counts().index)
# plt.show()

# Survival percentage based on gender
# plt.bar(np.arange(len(pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean())), pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean()*100)
# plt.xticks(np.arange(len(pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean())), pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().index)
# plt.ylabel('% survival')
# plt.show()