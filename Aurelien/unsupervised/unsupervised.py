import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

### K means
# blob_centers = np.array(
#     [[ 0.2,  2.3],
#      [-1.5 ,  2.3],
#      [-2.8,  1.8],
#      [-2.8,  2.8],
#      [-2.8,  1.3]])
# blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
# X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7)

# def plot_clusters(X, y=None):
#     plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
#     plt.xlabel("$x_1$", fontsize=14)
#     plt.ylabel("$x_2$", fontsize=14, rotation=0)
# plt.figure(figsize=(8, 4))
# plot_clusters(X)
# plt.show()

# k = 5
# kmeans = KMeans(n_clusters=k, random_state=42)
# y_pred = kmeans.fit_predict(X)
# print(kmeans.cluster_centers_)
# [[ 0.20876306  2.25551336]
#  [-2.80389616  1.80117999]
#  [-1.46679593  2.28585348]
#  [-2.79290307  2.79641063]
#  [-2.80037642  1.30082566]]
# plt.figure(figsize=(8, 4))
# plot_clusters(X)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, linewidths=8, zorder=10, alpha=0.9)
# plt.show()
# X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
# print(kmeans.predict(X_new))
# [0 0 3 3]
# print( kmeans.transform(X_new)) # distance from the cluster centers
# [[0.32995317 2.81093633 1.49439034 2.9042344  2.88633901]
#  [2.80290755 5.80730058 4.4759332  5.84739223 5.84236351]
#  [3.29399768 1.21475352 1.69136631 0.29040966 1.71086031]
#  [3.21806371 0.72581411 1.54808703 0.36159148 1.21567622]]
# print(kmeans.inertia_) #mean squared distance between each instance and its closest centroid
# 211.5985372581684
# print( kmeans.score(X))
# -211.59853725816845
# print(silhouette_score(X,kmeans.labels_))
# 0.6555176425728279

# image segmentation
# image = imread('D:\\tensor\\Aurelien\\unsupervised\\ladybug.png')
# print(image.shape)
# # (533, 800, 3)
# X = image.reshape(-1, 3)
# kmeans = KMeans(n_clusters=8).fit(X)
# segmented_img = kmeans.cluster_centers_[kmeans.labels_]
# segmented_img = segmented_img.reshape(image.shape)
# plt.imshow(segmented_img)
# plt.show()

# Preprocessing

# X_digits, y_digits = load_digits(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
# print(X_train.shape)
# (1347, 64)

# log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
# log_reg.fit(X_train, y_train)
# print(log_reg.score(X_test, y_test))
# # 0.9688888888888889
# pipeline = Pipeline([
#     ("kmeans", KMeans(n_clusters=50, random_state=42)),
#     ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
# ])
# pipeline.fit(X_train, y_train)
# print(pipeline.score(X_test, y_test))
# # 0.98

# param_grid = dict(kmeans__n_clusters=range(2, 100))
# grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2, n_jobs = -1)
# grid_clf.fit(X_train, y_train)
# print(grid_clf.best_params_)
# # {'kmeans__n_clusters': 57}
# print(grid_clf.score(X_test, y_test))
# # 0.98

# Semi-Supervised
# n_labeled = 50
# log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)
# log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
# print(log_reg.score(X_test, y_test))
# 0.8333333333333334
# k = 50
# kmeans = KMeans(n_clusters=k, random_state=42)
# X_digits_dist = kmeans.fit_transform(X_train)
# representative_digit_idx = np.argmin(X_digits_dist, axis=0)
# X_representative_digits = X_train[representative_digit_idx]
# plt.figure(figsize=(8, 2))
# for index, X_representative_digit in enumerate(X_representative_digits):
#     plt.subplot(k // 10, 10, index + 1)
#     plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
#     plt.axis('off')
# plt.show()
# y_representative_digits = np.array([0, 1, 3, 2, 7, 6, 4, 6, 9, 5,
#                                     1, 2, 9, 5, 2, 7, 8, 1, 8, 6,
#                                     3, 2, 5, 4, 5, 4, 0, 3, 2, 6,
#                                     1, 7, 7, 9, 1, 8, 6, 5, 4, 8,
#                                     5, 3, 3, 6, 7, 9, 7, 8, 4, 9])
# log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
# log_reg.fit(X_representative_digits, y_representative_digits)
# print(log_reg.score(X_test, y_test))
# 0.9133333333333333

# label propagation
# y_train_propagated = np.empty(len(X_train), dtype=np.int32)
# for i in range(k):
#     y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
# log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
# log_reg.fit(X_train, y_train_propagated)
# print(log_reg.score(X_test, y_test))
# 0.9244444444444444

### DBSCAN

# X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
# dbscan = DBSCAN(eps=0.05, min_samples=5)
# dbscan.fit(X)
# print(dbscan.labels_[:10])
# # [ 0  2 -1 -1  1  0  0  0  2  5]
# print(len(dbscan.core_sample_indices_))
# # 808
# print(dbscan.core_sample_indices_[:10])
# # [ 0  4  5  6  7  8 10 11 12 13]
# print(dbscan.components_[:3])
# # [[-0.02137124  0.40618608]
# #  [-0.84192557  0.53058695]
# #  [ 0.58930337 -0.32137599]]
# print(np.unique(dbscan.labels_))
# # [-1  0  1  2  3  4  5  6]

# dbscan = DBSCAN(eps=0.2, min_samples=5)
# dbscan.fit(X)
# cores = dbscan.components_
# core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
# core_mask[dbscan.core_sample_indices_] = True
# plt.scatter(cores[:, 0], cores[:, 1], c=dbscan.labels_[core_mask], s = 100)
# plt.scatter(X[:, 0], X[:, 1], s=10)
# plt.show()

### Gaussian Mixture
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]
# plt.scatter(X[:, 0], X[:, 1], s=10)
# plt.show()

# gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
# gm.fit(X)
# print(gm.weights_) # same as number blobs in X1 (500 + 500) and X2 (250)
# [0.39032584 0.20961444 0.40005972]
# print(gm.means_) # same centers
# [[ 0.05145113  0.07534576]
#  [ 3.39947665  1.05931088]
#  [-1.40764129  1.42712848]]
# print(gm.covariances_)
# [[[ 0.68825143  0.79617956]
#   [ 0.79617956  1.21242183]]
#
#  [[ 1.14740131 -0.03271106]
#   [-0.03271106  0.95498333]]
#
#  [[ 0.63478217  0.72970097]
#   [ 0.72970097  1.16094925]]]
# print(gm.converged_)
# True
# print(gm.n_iter_)
# 4
# print(gm.predict(X[:2]))
# [0 0]
# print(gm.predict_proba(X[:2]))
# [[9.76815996e-01 2.31833274e-02 6.76282339e-07]
#  [9.82914418e-01 1.64110061e-02 6.74575575e-04]]

# Bayesian Gaussian Mixture Models
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
print(np.round(bgm.weights_, 2)) # unnecessary clusters gets zero weight
# [0.4  0.   0.   0.   0.39 0.2  0.   0.   0.   0.  ]