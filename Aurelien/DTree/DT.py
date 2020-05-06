from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

# Classify
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target
# tree_clf = DecisionTreeClassifier(max_depth=2)
# tree_clf.fit(X, y)
# print(tree_clf.predict_proba([[5, 1.5]]))
# [[0.         0.90740741 0.09259259]]
# print( tree_clf.predict([[5, 1.5]]))
# [1]

# regres
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
