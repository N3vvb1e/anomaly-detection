import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

rng = np.random.default_rng(42)

def gen_train_2d(n_per_cluster=500, std=1.0, seed=42):
    X, _ = make_blobs(
        n_samples=2*n_per_cluster,
        centers=[(10.0, 0.0), (0.0, 10.0)],
        n_features=2,
        cluster_std=std,
        random_state=seed
    )
    return X

def gen_test_2d(n=1000, low=-10.0, high=20.0, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=(n, 2))

def gen_train_3d(n_per_cluster=500, std=1.0, seed=42):
    X, _ = make_blobs(
        n_samples=2*n_per_cluster,
        centers=[(0.0, 10.0, 0.0), (10.0, 0.0, 10.0)],
        n_features=3,
        cluster_std=std,
        random_state=seed
    )
    return X

def gen_test_3d(n=1000, low=-10.0, high=20.0, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(low=low, high=high, size=(n, 3))

def fit_models_2d(X_train, contamination=0.02, dif_hidden=(32, 16), loda_bins=30, seed=42):
    iforest = IForest(contamination=contamination, random_state=seed)
    iforest.fit(X_train)

    try:
        dif = DIF(contamination=contamination, hidden_neurons=dif_hidden, random_state=seed)
    except TypeError:
        dif = DIF(contamination=contamination, random_state=seed)
    dif.fit(X_train)

    loda = LODA(contamination=contamination, n_bins=loda_bins)
    loda.fit(X_train)

    return iforest, dif, loda

def plot_2d_scores(X_test, models, titles):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, model, title in zip(axes, models, titles):
        s = model.decision_function(X_test)
        sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=s, s=18)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.grid(True, alpha=0.2)
        plt.colorbar(sc, ax=ax, label="Outlier score")
    plt.show()

def plot_3d_scores(X_test, models, titles):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(15, 4.8))
    for i, (model, title) in enumerate(zip(models, titles), start=1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        s = model.decision_function(X_test)
        sc = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=s, s=8)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.7, label="Outlier score")
    plt.tight_layout(); plt.show()

X_train_2d = gen_train_2d()
X_test_2d = gen_test_2d()

dif_hidden = (64, 32)
loda_bins  = 20

iforest2d, dif2d, loda2d = fit_models_2d(
    X_train_2d, contamination=0.02, dif_hidden=dif_hidden, loda_bins=loda_bins, seed=42
)
plot_2d_scores(
    X_test_2d,
    models=[iforest2d, dif2d, loda2d],
    titles=["IForest (axis-parallel)", "DIF (Deep Isolation Forest)", "LODA (random proj. + hist)"]
)

X_train_3d = gen_train_3d()
X_test_3d = gen_test_3d()

iforest3d = IForest(contamination=0.02, random_state=42).fit(X_train_3d)
try:
    dif3d = DIF(contamination=0.02, hidden_neurons=dif_hidden, random_state=42).fit(X_train_3d)
except TypeError:
    dif3d = DIF(contamination=0.02, random_state=42).fit(X_train_3d)

loda3d = LODA(contamination=0.02, n_bins=loda_bins).fit(X_train_3d)

plot_3d_scores(
    X_test_3d,
    models=[iforest3d, dif3d, loda3d],
    titles=["IForest (3D)", "DIF (3D)", "LODA (3D)"]
)
