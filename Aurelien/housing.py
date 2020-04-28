import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

housing = pd.read_csv('D:\\tensor\\Aurelien\\housing.csv')
# print(housing.head())
# print(housing.info())
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

# # function to split data into train and test sets
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set))
# print(len(test_set))

# using sklearn to split data into train and test sets

# purely random
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(len(train_set))
# print(len(test_set))

#  stratified sampling based on the income category
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
# housing["income_cat"].hist()
# plt.show()
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
# print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))

# remove income cat col
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

















