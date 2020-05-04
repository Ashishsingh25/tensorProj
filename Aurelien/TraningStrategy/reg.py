import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)

### Normal Equation
# X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_best)
# [[4.17707456]
#  [2.87522069]]

# X_new = np.array([[0], [2]])
# X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
# y_predict = X_new_b.dot(theta_best)
# print(y_predict)
# [[4.12423027]
#  [9.73424972]]

# plt.plot(X_new, y_predict, "r-")
# plt.plot(X, y, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()

### using scikit learn
# lin_reg = LinearRegression()
# lin_reg.fit(X, y)
# print(lin_reg.intercept_, lin_reg.coef_)
# [4.24649833] [[2.87656222]]
# print(lin_reg.predict(X_new))
# [[4.24649833]
#  [9.99962277]]

### least squares using SVD pseudoinverse
# theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
# print(theta_best_svd)
# [[4.1914519 ]
#  [2.84543135]]
# print(np.linalg.pinv(X_b).dot(y))
# [[4.1914519 ]
#  [2.84543135]]

### Batch GD
# eta = 0.1 # learning rate
# n_iterations = 1000
# m = 100
# theta = np.random.randn(2,1) # random initialization
# for iteration in range(n_iterations):
#     gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
#     theta = theta - eta * gradients
# print(theta)
# [[4.32530102]
#  [2.80622989]]

### Stochastic Gradient Descent

# n_epochs = 50
# t0, t1 = 5, 50 # learning schedule hyperparameters
# def learning_schedule(t):
#     return t0 / (t + t1)
# theta = np.random.randn(2,1) # random initialization
# for epoch in range(n_epochs):
#     for i in range(m):
#         random_index = np.random.randint(m)
#         xi = X_b[random_index:random_index+1]
#         yi = y[random_index:random_index+1]
#         gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
#         eta = learning_schedule(epoch * m + i)
#         theta = theta - eta * gradients
# print(theta)
# [[4.19615251]
#  [2.95095465]]

### SGD using scikit learn
# sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
# sgd_reg.fit(X, y.ravel())
# print( sgd_reg.intercept_, sgd_reg.coef_)
# [4.19887984] [2.92796386]

### Polynomial Regression
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# using sk learn
# poly_features = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly_features.fit_transform(X)
# print(X[0])
# [2.50674277]
# print(X_poly[0])
# [2.50674277 6.2837593 ] ## Original data and sq of the data

# lin_reg = LinearRegression()
# lin_reg.fit(X_poly, y)
# print(lin_reg.intercept_, lin_reg.coef_)
# [1.73380095] [[0.98101307 0.56276813]]

### Regularized Linear Models

# Ridge
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
# ridge_reg = Ridge(alpha=1, solver="cholesky")
# ridge_reg.fit(X, y)
# print(ridge_reg.predict([[1.5]]))
# [[4.67832751]]
# sgd_reg = SGDRegressor(penalty="l2")
# sgd_reg.fit(X, y.ravel())
# print(sgd_reg.predict([[1.5]]))
# [4.63049333]

# Lasso
# lasso_reg = Lasso(alpha=0.1)
# lasso_reg.fit(X, y)
# print(lasso_reg.predict([[1.5]]))
# [4.61955096]

# ElasticNet
# elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
# elastic_net.fit(X, y)
# print(elastic_net.predict([[1.5]]))
# [4.62260351]

### Logistic Regression
iris = datasets.load_iris()
# print(list(iris.keys()))
# ['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']
# X = iris["data"][:, 3:] # petal width
# print(X[:10])
# y = (iris["target"] == 2).astype(np.int) # 1 if Iris-Virginica, else 0
# print(y[:10])

# log_reg = LogisticRegression()
# log_reg.fit(X, y)

# X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# y_proba = log_reg.predict_proba(X_new)
# plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
# plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
# plt.show()

# print(log_reg.predict([[1.7], [1.5]]))
# [1 0]

### Multinomial Logistic Regression
# X = iris["data"][:, (2, 3)] # petal length, petal width
# y = iris["target"]
# softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
# softmax_reg.fit(X, y)
# print(softmax_reg.predict([[5, 2]]))
# [2]
# print(softmax_reg.predict_proba([[5, 2]]))
# [[6.38014896e-07 5.74929995e-02 9.42506362e-01]]



















