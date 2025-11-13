import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

rng = np.random.default_rng(42)

X_train, _ = make_blobs(
    n_samples=500,
    centers=[(0.0, 0.0)],
    n_features=2,
    cluster_std=1.0,
    random_state=42
)

def random_unit_vectors(d=2, k=5, rng=rng):
    V = rng.normal(size=(k, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V

proj_vecs = random_unit_vectors(d=2, k=5, rng=rng)

def build_histograms(X, proj_vecs, n_bins=30, k_std=5.0):
    hists = []
    for v in proj_vecs:
        z = X @ v
        m, s = z.mean(), z.std()
        lo, hi = m - k_std * s, m + k_std * s
        counts, edges = np.histogram(z, bins=n_bins, range=(lo, hi))
        probs = counts.astype(float) / max(counts.sum(), 1)     # avoid div by 0
        hists.append((edges, probs))
    return hists

def mean_bin_probability(X, proj_vecs, hists, eps=1e-12):
    n = X.shape[0]
    k = len(proj_vecs)
    probs = np.zeros((n, k), dtype=float)

    for j, (v, (edges, p)) in enumerate(zip(proj_vecs, hists)):
        z = X @ v
        idx = np.searchsorted(edges, z, side="right") - 1
        idx = np.clip(idx, 0, len(p) - 1)
        probs[:, j] = p[idx] + eps      # tiny floor to avoid exact zeros

    return probs.mean(axis=1)

X_test = rng.uniform(low=-3.0, high=3.0, size=(500, 2))

def plot_for_bins_list(bins_list=(10, 25, 50), view="probability"):
    """
    view: "probability" (spec-compliant) or "anomaly" (1 - probability) for visuals.
    """
    n_cols = len(bins_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5*n_cols, 4.5), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    for ax, n_bins in zip(axes, bins_list):
        hists = build_histograms(X_train, proj_vecs, n_bins=n_bins, k_std=5.0)
        score = mean_bin_probability(X_test, proj_vecs, hists)      # mean bin prob

        if view == "anomaly":
            values = 1.0 - score
            label = "1 - mean bin probability"
            title = f"Anomaly-colored map (bins={n_bins})"
        else:
            values = score
            label = "Mean bin probability"
            title = f"Score map (bins={n_bins})"

        sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=values, s=25)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.2)
        fig.colorbar(sc, ax=ax, label=label)

    plt.show()

plot_for_bins_list(bins_list=(10, 25, 50), view="probability")
# plot_for_bins_list(bins_list=(10, 25, 50), view="anomaly")