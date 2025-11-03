import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import balanced_accuracy_score

from pyod.models.knn import KNN
from pyod.utils.data import generate_data_clusters

RNG = 42

X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=400,
    n_test=200,
    n_features=2,
    n_clusters=2,
    contamination=0.10,
    random_state=RNG,
)

k_list = [1, 5, 15, 30]
results = []

fig, axes = plt.subplots(len(k_list), 4, figsize=(16, 4 * len(k_list)), squeeze=False)

for i, k in enumerate(k_list):
    clf = KNN(n_neighbors=k, contamination=0.10)
    clf.fit(X_train)

    y_pred_train = clf.labels_              
    y_pred_test  = clf.predict(X_test)      

    bacc_train = balanced_accuracy_score(y_train, y_pred_train)
    bacc_test  = balanced_accuracy_score(y_test,  y_pred_test)
    results.append((k, bacc_train, bacc_test))

    ax = axes[i, 0]
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=18, cmap="coolwarm", edgecolor="none")
    ax.set_title(f"k={k}  •  Ground truth (train)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(alpha=.25)

    ax = axes[i, 1]
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train, s=18, cmap="coolwarm", edgecolor="none")
    ax.set_title(f"k={k}  •  Predicted (train)\nBalanced acc: {bacc_train:.3f}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(alpha=.25)

    ax = axes[i, 2]
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=18, cmap="coolwarm", edgecolor="none")
    ax.set_title(f"k={k}  •  Ground truth (test)")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(alpha=.25)

    ax = axes[i, 3]
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, s=18, cmap="coolwarm", edgecolor="none")
    ax.set_title(f"k={k}  •  Predicted (test)\nBalanced acc: {bacc_test:.3f}")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.grid(alpha=.25)

legend_elems = [
    Line2D([0], [0], marker='o', color='w', label='inlier (0)', markerfacecolor='navy', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='outlier (1)', markerfacecolor='red', markersize=8),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()

print("Balanced accuracy per k (train, test):")
for k, bt, be in results:
    print(f"k={k:>2}  |  train={bt:.3f}  test={be:.3f}")
