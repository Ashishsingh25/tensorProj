import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


### diverse set of classifiers
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# log_clf = LogisticRegression(solver="lbfgs", random_state=42)
# rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
# svm_clf = SVC(gamma="scale", random_state=42)
# voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
# voting_clf.fit(X_train, y_train)
# for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
# LogisticRegression 0.864
# RandomForestClassifier 0.896
# SVC 0.896
# VotingClassifier 0.912

### Bagging

# bag_clf = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=500,
#                             max_samples=100, bootstrap=True, random_state=42)
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))
# 0.904
# single Decision Tree Classifier
# tree_clf = DecisionTreeClassifier(random_state=42)
# tree_clf.fit(X_train, y_train)
# y_pred_tree = tree_clf.predict(X_test)
# print(accuracy_score(y_test, y_pred_tree))
# 0.856

### Random Forest

# bag_clf = BaggingClassifier(DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
#                             n_estimators=500, max_samples=1.0, bootstrap=True, random_state=42)
# bag_clf.fit(X_train, y_train)
# y_pred = bag_clf.predict(X_test)

# rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
# rnd_clf.fit(X_train, y_train)
# y_pred_rf = rnd_clf.predict(X_test)
# Comparing Bagging with Forest
# print(np.sum(y_pred == y_pred_rf) / len(y_pred))
# 0.976 # almost the same

# feature importance
# iris = load_iris()
# rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
# rnd_clf.fit(iris["data"], iris["target"])
# for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
#     print(name, score)
# sepal length (cm) 0.09160374511641181
# sepal width (cm) 0.021656169338862472
# petal length (cm) 0.41171839270688726
# petal width (cm) 0.47502169283783857

### Boosting

# adaptive boosting
# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100,
#                              algorithm="SAMME.R", learning_rate=0.5, random_state=42)
# ada_clf.fit(X_train, y_train)
# y_pred_ada = ada_clf.predict(X_test)
# print(accuracy_score(y_test, y_pred_ada))
# 0.912

# Gradient Boosting
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

# tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
# tree_reg1.fit(X, y)
# print(y[15])
# # 0.3491303626700779
# print(tree_reg1.predict(X[[15]]))
# # [0.12356613]
# y2 = y - tree_reg1.predict(X)
# tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
# tree_reg2.fit(X, y2)
# print(tree_reg1.predict(X[[15]]) + tree_reg2.predict(X[[15]]))
# # [0.28340415]
# y3 = y2 - tree_reg2.predict(X)
# tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
# tree_reg3.fit(X, y3)
# print(tree_reg1.predict(X[[15]]) + tree_reg2.predict(X[[15]]) + tree_reg3.predict(X[[15]]))
# # [0.29044761]

# Above can be done by using GradientBoostingRegressor in scikit learn
# gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
# gbrt.fit(X, y)
# print(gbrt.predict(X[[15]]))
# # [0.29044761]

# early stopping

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

# training a finite large number of estimators (n_estimators=120)
# gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
# gbrt.fit(X_train, y_train)
# errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
# bst_n_estimators = np.argmin(errors) + 1
# print(bst_n_estimators)
# # 56
# gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
# gbrt_best.fit(X_train, y_train)
# print(gbrt_best.predict(X[[15]]))
# [0.34657576]
# print(y[15])
# # 0.3491303626700779

# stopping training early when error does not improve
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)
min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # early stopping
print(gbrt.n_estimators - 5)
# 56