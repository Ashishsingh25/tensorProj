import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
from sklearn.metrics import mean_squared_error

### 3d dataset
# np.random.seed(4)
# m = 60
# w1, w2 = 0.1, 0.3
# noise = 0.1
# angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
# X = np.empty((m, 3))
# X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
# X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
# X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

### PCA using SVD
# X_centered = X - X.mean(axis=0)
# U, s, Vt = np.linalg.svd(X_centered)
# c1 = Vt.T[:, 0]
# c2 = Vt.T[:, 1]
# W2 = Vt.T[:, :2]
# X2D_using_svd = X_centered.dot(W2)
# print(X2D_using_svd[:5])
# [[-1.26203346 -0.42067648]
#  [ 0.08001485  0.35272239]
#  [-1.17545763 -0.36085729]
#  [-0.89305601  0.30862856]
#  [-0.73016287  0.25404049]]

### PCA using sk learn
# pca = PCA(n_components = 2)
# X2D = pca.fit_transform(X)
# print(X2D[:5])
# [[ 1.26203346  0.42067648]
#  [-0.08001485 -0.35272239]
#  [ 1.17545763  0.36085729]
#  [ 0.89305601 -0.30862856]
#  [ 0.73016287 -0.25404049]]
# print(pca.explained_variance_ratio_)
# [0.84248607 0.14631839]

# finding correct number of dim to explain 95% variance
# mnist = fetch_openml('mnist_784', version=1)
# mnist.target = mnist.target.astype(np.uint8)
# X = mnist["data"]
# y = mnist["target"]
# X_train, X_test, y_train, y_test = train_test_split(X, y)

# pca = PCA()
# pca.fit(X_train)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1
# print(d)
# 154

# OR
# pca = PCA(n_components=0.95)
# X_reduced = pca.fit_transform(X_train)
# print(pca.n_components_)
# # 154
# print(np.sum(pca.explained_variance_ratio_))
# # 0.9503928114930393

# OR
# plt.figure(figsize=(6,4))
# plt.plot(cumsum, linewidth=3)
# plt.axis([0, 400, 0, 1])
# plt.xlabel("Dimensions")
# plt.ylabel("Explained Variance")
# plt.grid(True)
# plt.show()

### Kernel PCA

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
y = t > 6.9
# clf = Pipeline([
#         ("kpca", KernelPCA(n_components=2)),
#         ("log_reg", LogisticRegression(solver="lbfgs"))
#     ])
#
# param_grid = [{
#         "kpca__gamma": np.linspace(0.03, 0.05, 10),
#         "kpca__kernel": ["rbf", "sigmoid"]
#     }]
#
# grid_search = GridSearchCV(clf, param_grid, cv=3, verbose= 2, n_jobs= -1)
# grid_search.fit(X, y)
# print(grid_search.best_params_)
# {'kpca__gamma': 0.043333333333333335, 'kpca__kernel': 'rbf'}
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
                    fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)
print(mean_squared_error(X, X_preimage))
# 32.78630879576613



