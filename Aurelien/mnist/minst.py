from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_openml('mnist_784', version=1)
# print(mnist.keys())
# dict_keys(['data', 'target', 'frame', 'feature_names', 'target_names', 'DESCR', 'details', 'categories', 'url'])

X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)
# print(X.shape)
# (70000, 784)
# print(y.shape)
# (70000,)

# plotting the 1st instance
some_digit = X[0]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()
# print(y[0])
# 5

# splitting into training and test
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

##### Binary Classifier
# y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
# y_test_5 = (y_test == 5)

sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)
# print(sgd_clf.predict([some_digit]))
# [ True]

### Cross - Validation
# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
# [0.95035 0.96035 0.9604 ]

# classify all instances as not 5
# class Never5Classifier(BaseEstimator):
#     def fit(self, X, y=None):
#         pass
#
#     def predict(self, X):
#         return np.zeros((len(X), 1), dtype=bool)
#
# never_5_clf = Never5Classifier()
# print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
# [0.91125 0.90855 0.90915]

### Confusion Matrix
# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# print(confusion_matrix(y_train_5, y_train_pred))
# [[53892   687]
#  [ 1891  3530]]

# y_train_perfect_predictions = y_train_5 # pretend we reached perfection
# print(confusion_matrix(y_train_5, y_train_perfect_predictions))
# [[54579     0]
#  [    0  5421]]

# print(precision_score(y_train_5, y_train_pred))  # == 3530 / (3530 + 687)
# 0.8370879772350012
# print(recall_score(y_train_5, y_train_pred)) # == 3530 / (3530 + 1891)
# 0.6511713705958311
# print(f1_score(y_train_5, y_train_pred))
# 0.7325171197343846

# print(sgd_clf.decision_function([some_digit]))

# y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
# precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
# plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
# plt.show()

# threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
# y_train_pred_90 = (y_scores >= threshold_90_precision)
# print(precision_score(y_train_5, y_train_pred_90))
# 0.9000345901072293
# print(recall_score(y_train_5, y_train_pred_90))
# 0.4799852425751706

### ROC curve

# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal

# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# plot_roc_curve(fpr, tpr)
# plt.show()

# print(roc_auc_score(y_train_5, y_scores))
# 0.9604938554008616

# comparing to Random forest
forest_clf = RandomForestClassifier(random_state=42)
# y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
# y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
# fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
#
# plt.plot(fpr, tpr, "b:", label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="lower right")
# plt.show()
# print(roc_auc_score(y_train_5, y_scores_forest))
# 0.9983436731328145

##### Multiclass Classication

# one-versus-all
# sgd_clf.fit(X_train, y_train) # y_train, not y_train_5
# print(sgd_clf.predict([some_digit]))
# [3]
# some_digit_scores = sgd_clf.decision_function([some_digit])
# print(some_digit_scores)
# [[-31893.03095419 -34419.69069632  -9530.63950739   1823.73154031
#   -22320.14822878  -1385.80478895 -26188.91070951 -16147.51323997
#    -4604.35491274 -12050.767298  ]]
# print(np.argmax(some_digit_scores))
# 3
# print(sgd_clf.classes_)
# [0 1 2 3 4 5 6 7 8 9]
# print(sgd_clf.classes_[np.argmax(some_digit_scores)])
# 3
# print(len(sgd_clf.estimators_))

# one-versus-one

# ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
# ovo_clf.fit(X_train, y_train)
# print(ovo_clf.predict([some_digit]))
# [5]
# print(len(ovo_clf.estimators_))
# 45

# random forest can classify multinomial data

# forest_clf.fit(X_train, y_train)
# print(forest_clf.predict([some_digit]))
# [5]
# print(forest_clf.predict_proba([some_digit]))
# [[0.   0.   0.01 0.08 0.   0.9  0.   0.   0.   0.01]]

# cross validation

# print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
# [0.87365 0.85835 0.8689 ]

# standardization

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
# [0.8983 0.891  0.9018]

# confusion matrix

# y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
# conf_mx = confusion_matrix(y_train, y_train_pred)
# print(conf_mx)
# [[5577    0   22    5    8   43   36    6  225    1]
#  [   0 6400   37   24    4   44    4    7  212   10]
#  [  27   27 5220   92   73   27   67   36  378   11]
#  [  22   17  117 5227    2  203   27   40  403   73]
#  [  12   14   41    9 5182   12   34   27  347  164]
#  [  27   15   30  168   53 4444   75   14  535   60]
#  [  30   15   42    3   44   97 5552    3  131    1]
#  [  21   10   51   30   49   12    3 5684  195  210]
#  [  17   63   48   86    3  126   25   10 5429   44]
#  [  25   18   30   64  118   36    1  179  371 5107]]
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

# error matrix
# row_sums = conf_mx.sum(axis=1, keepdims=True)
# norm_conf_mx = conf_mx / row_sums
# np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()

##### Multilabel Classification

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print(knn_clf.predict([some_digit]))
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
print(f1_score(y_multilabel, y_train_knn_pred, average="macro"))
# [[False  True]]

