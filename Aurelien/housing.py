import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats
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

###  stratified sampling based on the income category
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

housing = strat_train_set.copy()

#  Visualize
# housing.plot(kind = 'scatter', x= "longitude", y= "latitude", alpha=0.1)
# plt.show()
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=housing["population"]/100, label="population", figsize=(10,7),
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
# plt.legend()
# plt.show()

# Correlations
# corr_matrix = housing.corr()
# print(corr_matrix['median_house_value'].sort_values(ascending=False))

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"]=housing["population"]/housing["households"]
# corr_matrix = housing.corr()
# print(corr_matrix['median_house_value'].sort_values(ascending=False))

### Prepare the Data for ML

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# housing.dropna(subset=["total_bedrooms"]) # to remove the rows with NA
# housing.drop("total_bedrooms", axis=1) # remove the col
# median = housing["total_bedrooms"].median() # replace NA with median
# housing["total_bedrooms"].fillna(median, inplace=True)

# using Imputer to replace NA values
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
# print(imputer.statistics_)
# print(housing_num.median().values)
X = imputer.transform(housing_num) # Numpy array
housing_tr = pd.DataFrame(X, columns=housing_num.columns) # pd dataframe

# Categorical Values
housing_cat = housing[["ocean_proximity"]]
# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat.head(10))
# print(housing_cat_encoded[:10])
# print(ordinal_encoder.categories_)
# this results in a numerical encoding which might mean 0 (<1H OCEAN) is close to 1 (INLAND)

# using one hot encoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(type(housing_cat_1hot)) #scipy.sparse.csr.csr_matrix
# print(housing_cat.head(10))
# print(housing_cat_1hot[:10]) # only stores location of non zero elements
# print( housing_cat_1hot.toarray()[:10])
# print(cat_encoder.categories_)

# Custom Transformers

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)
# print(housing_extra_attribs[:10])

# using Pipelines

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())])
# housing_num_tr = num_pipeline.fit_transform(housing_num)
# print(housing_num_tr[:10])

num_attribs = list(housing_num)
# print(num_attribs)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)])
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing[:10])
# print(housing_prepared[:10])

### Train a Model

# Simple model - Linear Reg
# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)

# testing for some data
# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions:", lin_reg.predict(some_data_prepared))
# print("Labels:", list(some_labels))

# RMSE
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse) # shows underfitting

# More complex model Decision Tree Regressor
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse) # Overfitting

# Cross - Validation
# scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)
# def display_scores(scores):
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())
# print('tree_reg')
# display_scores(tree_rmse_scores)

# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# print('lin_reg')
# display_scores(lin_rmse_scores)

# Random Forest Regressor
# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# housing_predictions = forest_reg.predict(housing_prepared)
# forest_mse = mean_squared_error(housing_labels, housing_predictions)
# forest_rmse = np.sqrt(forest_mse)
# print(forest_rmse)

# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)

### Fine-Tune Model

# Grid Search
param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
              {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print( grid_search.best_params_)
# {'max_features': 6, 'n_estimators': 30}
print( grid_search.best_estimator_)
# RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
#                       max_depth=None, max_features=6, max_leaf_nodes=None,
#                       max_samples=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=1,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       n_estimators=30, n_jobs=None, oob_score=False,
#                       random_state=None, verbose=0, warm_start=False)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
# 64958.07313746577 {'max_features': 2, 'n_estimators': 3}
# 55606.95406660094 {'max_features': 2, 'n_estimators': 10}
# 52919.77847250474 {'max_features': 2, 'n_estimators': 30}
# 61030.60952235604 {'max_features': 4, 'n_estimators': 3}
# 53051.75350457917 {'max_features': 4, 'n_estimators': 10}
# 50426.48474898194 {'max_features': 4, 'n_estimators': 30}
# 58599.48202606033 {'max_features': 6, 'n_estimators': 3}
# 52127.61532130916 {'max_features': 6, 'n_estimators': 10}
# 49913.78438595976 {'max_features': 6, 'n_estimators': 30}
# 58876.80116214216 {'max_features': 8, 'n_estimators': 3}
# 51927.829198125306 {'max_features': 8, 'n_estimators': 10}
# 50315.41428596045 {'max_features': 8, 'n_estimators': 30}
# 62106.668237644284 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
# 54886.75253435984 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
# 60833.507419788 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
# 52374.31089510191 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
# 58895.69356269152 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
# 51828.44779354884 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}

# attribute score
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))
# [(0.34813721923679297, 'median_income'),
#  (0.16869871179013715, 'INLAND'),
#  (0.11707764685735152, 'pop_per_hhold'),
#  (0.08137045181321519, 'bedrooms_per_room'),
#  (0.07008835726286751, 'longitude'),
#  (0.06106235862067891, 'latitude'),
#  (0.042840489262163356, 'housing_median_age'),
#  (0.0387495698908087, 'rooms_per_hhold'),
#  (0.015338733575312725, 'population'),
#  (0.015020453466820181, 'total_rooms'),
#  (0.014509687724417027, 'total_bedrooms'),
#  (0.014349787946886843, 'households'),
#  (0.006200408231224715, '<1H OCEAN'),
#  (0.004427805092308678, 'NEAR OCEAN'),
#  (0.0019994444552519095, 'NEAR BAY'),
#  (0.00012887477376266347, 'ISLAND')]

# performance on test set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
# 48332.858448770654

# Error confidence interval
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
print(np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors))))
# [46360.42505503 50227.89464406]















