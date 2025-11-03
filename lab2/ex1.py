import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

rng = np.random.default_rng(42)

def hat_diag(X):
    """Return leverage scores (diag of hat matrix) for design X (with intercept column included)."""
    XtX = X.T @ X
    H = X @ np.linalg.inv(XtX) @ X.T
    return np.diag(H)

def make_groups_1d(a=2.0, b=0.5, mu=0.0, sigma=0.5,
                   n_each=50, 
                   sigma_x_hi=3.0, sigma_y_hi=3.0):
    """Create 4 groups of points for the 1D model."""
   
    x_r = rng.normal(0.0, 1.0, n_each)
    eps_r = rng.normal(mu, sigma, n_each)
    y_r = a * x_r + b + eps_r
    g_r = np.full(n_each, 0)

    x_hx = rng.normal(0.0, sigma_x_hi, n_each)
    eps_hx = rng.normal(mu, sigma, n_each)
    y_hx = a * x_hx + b + eps_hx
    g_hx = np.full(n_each, 1)

    x_hy = rng.normal(0.0, 1.0, n_each)
    eps_hy = rng.normal(mu, sigma_y_hi, n_each)
    y_hy = a * x_hy + b + eps_hy
    g_hy = np.full(n_each, 2)

    x_hb = rng.normal(0.0, sigma_x_hi, n_each)
    eps_hb = rng.normal(mu, sigma_y_hi, n_each)
    y_hb = a * x_hb + b + eps_hb
    g_hb = np.full(n_each, 3)

    x = np.concatenate([x_r,  x_hx,  x_hy,  x_hb])
    y = np.concatenate([y_r,  y_hx,  y_hy,  y_hb])
    grp = np.concatenate([g_r, g_hx, g_hy, g_hb])
    return x, y, grp

def make_groups_2d(a=1.0, b=1.5, c=0.2, mu=0.0, sigma=0.5,
                   n_each=60, sigma_x_hi=2.5, sigma_y_hi=2.5):
    """Create 4 groups for 2D: vary spread in predictors (x1,x2) and/or response noise."""
  
    x1_r = rng.normal(0.0, 1.0, n_each)
    x2_r = rng.normal(0.0, 1.0, n_each)
    eps_r = rng.normal(mu, sigma, n_each)
    y_r = a*x1_r + b*x2_r + c + eps_r
    g_r = np.full(n_each, 0)

    x1_hx = rng.normal(0.0, sigma_x_hi, n_each)
    x2_hx = rng.normal(0.0, sigma_x_hi, n_each)
    eps_hx = rng.normal(mu, sigma, n_each)
    y_hx = a*x1_hx + b*x2_hx + c + eps_hx
    g_hx = np.full(n_each, 1)

    x1_hy = rng.normal(0.0, 1.0, n_each)
    x2_hy = rng.normal(0.0, 1.0, n_each)
    eps_hy = rng.normal(mu, sigma_y_hi, n_each)
    y_hy = a*x1_hy + b*x2_hy + c + eps_hy
    g_hy = np.full(n_each, 2)

    x1_hb = rng.normal(0.0, sigma_x_hi, n_each)
    x2_hb = rng.normal(0.0, sigma_x_hi, n_each)
    eps_hb = rng.normal(mu, sigma_y_hi, n_each)
    y_hb = a*x1_hb + b*x2_hb + c + eps_hb
    g_hb = np.full(n_each, 3)

    x1 = np.concatenate([x1_r, x1_hx, x1_hy, x1_hb])
    x2 = np.concatenate([x2_r, x2_hx, x2_hy, x2_hb])
    y  = np.concatenate([y_r,  y_hx,  y_hy,  y_hb])
    grp = np.concatenate([g_r, g_hx, g_hy, g_hb])
    return x1, x2, y, grp

group_cmap = ListedColormap(["#4c78a8", "#f58518", "#54a24b", "#e45756"])  # reg, hx, hy, both

def plot_grid_1d(mus, sigmas, a=2.0, b=0.5, n_each=50, top_k=8):
    fig, axes = plt.subplots(len(mus), len(sigmas), figsize=(4*len(sigmas), 3.6*len(mus)), squeeze=False)
    for i, mu in enumerate(mus):
        for j, sigma in enumerate(sigmas):
            x, y, grp = make_groups_1d(a=a, b=b, mu=mu, sigma=sigma, n_each=n_each)
            X = np.c_[np.ones_like(x), x]  # intercept + x
            lev = hat_diag(X)

            ax = axes[i, j]
            sc = ax.scatter(x, y, c=grp, cmap=group_cmap, s=20, alpha=0.8, edgecolor="none")
            
            top_idx = np.argsort(lev)[-top_k:]
            ax.scatter(x[top_idx], y[top_idx], facecolors='none', edgecolors='k', s=120, linewidths=1.8, label=f"Top-{top_k} leverage")

            ax.set_title(rf"1D  $\mu={mu}$,  $\sigma^2={sigma**2:.2f}$")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend(loc="upper left", fontsize=8, frameon=True)
            ax.grid(alpha=0.25)

    handles = [plt.Line2D([0],[0], marker='o', color='w', label='regular', markerfacecolor=group_cmap(0), markersize=8),
               plt.Line2D([0],[0], marker='o', color='w', label='high-var x', markerfacecolor=group_cmap(1), markersize=8),
               plt.Line2D([0],[0], marker='o', color='w', label='high-var y', markerfacecolor=group_cmap(2), markersize=8),
               plt.Line2D([0],[0], marker='o', color='w', label='high-var both', markerfacecolor=group_cmap(3), markersize=8)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()

def plot_grid_2d(mus, sigmas, a=1.2, b=1.0, c=0.0, n_each=60, top_k=10):
    fig, axes = plt.subplots(len(mus), len(sigmas), figsize=(4*len(sigmas), 3.6*len(mus)), squeeze=False)
    for i, mu in enumerate(mus):
        for j, sigma in enumerate(sigmas):
            x1, x2, y, grp = make_groups_2d(a=a, b=b, c=c, mu=mu, sigma=sigma, n_each=n_each)
            # design with intercept + two predictors
            X = np.c_[np.ones_like(x1), x1, x2]
            lev = hat_diag(X)

            ax = axes[i, j]
            
            sizes = 10 + 140 * (lev - lev.min()) / (lev.max() - lev.min() + 1e-9)
            ax.scatter(x1, x2, c=grp, cmap=group_cmap, s=sizes, alpha=0.8, edgecolor="none")
            top_idx = np.argsort(lev)[-top_k:]
            ax.scatter(x1[top_idx], x2[top_idx], facecolors='none', edgecolors='k', s=160, linewidths=1.8, label=f"Top-{top_k} leverage")

            ax.set_title(rf"2D  $\mu={mu}$,  $\sigma^2={sigma**2:.2f}$")
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.legend(loc="upper left", fontsize=8, frameon=True)
            ax.grid(alpha=0.25)

    handles = [plt.Line2D([0],[0], marker='o', color='w', label='regular', markerfacecolor=group_cmap(0), markersize=8),
               plt.Line2D([0],[0], marker='o', color='w', label='high-var x (predictors)', markerfacecolor=group_cmap(1), markersize=8),
               plt.Line2D([0],[0], marker='o', color='w', label='high-var y (noise)', markerfacecolor=group_cmap(2), markersize=8),
               plt.Line2D([0],[0], marker='o', color='w', label='high-var both', markerfacecolor=group_cmap(3), markersize=8)]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()

mus = [0.0, 1.0]    # different noise means
sigmas = [0.3, 0.8, 1.5] 

plot_grid_1d(mus, sigmas, a=2.0, b=0.5, n_each=60, top_k=10)
plot_grid_2d(mus, sigmas, a=1.2, b=1.0, c=0.0, n_each=60, top_k=12)
