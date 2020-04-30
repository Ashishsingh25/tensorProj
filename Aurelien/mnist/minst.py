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

# Binary Classifier
y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

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

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
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

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# plot_roc_curve(fpr, tpr)
# plt.show()

print(roc_auc_score(y_train_5, y_scores))
# 0.9604938554008616

# comparing to Random forest
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()
print(roc_auc_score(y_train_5, y_scores_forest))
# 0.9983436731328145



