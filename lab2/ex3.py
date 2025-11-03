import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs

from pyod.models.knn import KNN
from pyod.models.lof import LOF

RNG = 42
cmap01 = ListedColormap(["navy", "red"]) 

X, y_true = make_blobs(
    n_samples=[200, 100],
    centers=[(-10, -10), (10, 10)],
    cluster_std=[2, 6],
    n_features=2,
    random_state=RNG
)

xlim = (X[:, 0].min() - 2, X[:, 0].max() + 2)
ylim = (X[:, 1].min() - 2, X[:, 1].max() + 2)

k_list = [5, 15, 30]

fig, axes = plt.subplots(len(k_list), 2, figsize=(10, 4 * len(k_list)), squeeze=False)

for i, k in enumerate(k_list):
    knn = KNN(n_neighbors=k, contamination=0.07)
    knn.fit(X)
    y_knn = knn.labels_

    lof = LOF(n_neighbors=k, contamination=0.07)
    lof.fit(X)
    y_lof = lof.labels_

    ax = axes[i, 0]
    ax.scatter(X[:, 0], X[:, 1], c=y_knn, s=20, cmap=cmap01, edgecolor="none")
    ax.set_title(f"KNN  (k={k})")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(alpha=.25)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

    ax = axes[i, 1]
    ax.scatter(X[:, 0], X[:, 1], c=y_lof, s=20, cmap=cmap01, edgecolor="none")
    ax.set_title(f"LOF  (k={k})")
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(alpha=.25)
    ax.set_xlabel("x1"); ax.set_ylabel("x2")

fig.suptitle("Ex. 3 — Different cluster densities: KNN vs LOF (red = outliers)", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()