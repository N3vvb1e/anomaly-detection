import os
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from sklearn.metrics import roc_auc_score

from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_scipy_sparse_matrix


def _as_csr(mat) -> sp.csr_matrix:
    if sp.issparse(mat):
        return mat.tocsr()
    return sp.csr_matrix(mat)

def _extract_label_vector(label_raw: np.ndarray) -> np.ndarray:
    y = np.asarray(label_raw)
    y = np.squeeze(y)

    if y.ndim == 2:
        if y.shape[0] < y.shape[1]:
            y = y.T
        if y.ndim == 2 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        else:
            y = np.squeeze(y)

    return np.asarray(y).reshape(-1)

def _binarize_for_auc(y: np.ndarray) -> np.ndarray:
    uniq, counts = np.unique(y, return_counts=True)
    if len(uniq) == 2 and set(uniq.tolist()) <= {0, 1}:
        return y.astype(int)

    anomaly_class = uniq[np.argmin(counts)]
    return (y == anomaly_class).astype(int)

def load_acm_mat_sparse(path: str):
    md = loadmat(path)
    X_raw = md["Attributes"]
    A_raw = md["Network"]
    y_raw = md["Label"]

    X_csr = _as_csr(X_raw)     # keep sparse
    A_csr = _as_csr(A_raw)     # keep sparse

    y_vec = _extract_label_vector(y_raw)
    y_bin = _binarize_for_auc(y_vec)

    return X_csr, A_csr, y_vec, y_bin

def induced_subgraph(X_csr: sp.csr_matrix,
                     A_csr: sp.csr_matrix,
                     y_vec: np.ndarray,
                     y_bin: np.ndarray,
                     max_nodes: int,
                     seed: int = 42):
    n = A_csr.shape[0]
    if max_nodes >= n:
        idx = np.arange(n, dtype=int)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_nodes, replace=False)
        idx = np.sort(idx)

    X_sub = X_csr[idx, :].tocsr()
    A_sub = A_csr[idx, :][:, idx].tocsr()
    y_sub = y_vec[idx]
    ybin_sub = y_bin[idx]

    return X_sub, A_sub, y_sub, ybin_sub, idx


class Encoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        z = F.relu(self.conv2(h, edge_index))
        return z  # [N, 64]


class AttributeDecoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.conv1 = GCNConv(64, 128)
        self.conv2 = GCNConv(128, out_dim)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(z, edge_index))
        x_hat = F.relu(self.conv2(h, edge_index))
        return x_hat  # [N, F]


class StructureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = GCNConv(64, 64)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv(z, edge_index))  # Z
        a_hat = h @ h.t()                     # Z @ Z^T  => [N, N]
        return a_hat


class GraphAutoencoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.encoder = Encoder(in_dim)
        self.attr_decoder = AttributeDecoder(out_dim=in_dim)
        self.struct_decoder = StructureDecoder()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        z = self.encoder(x, edge_index)
        x_hat = self.attr_decoder(z, edge_index)
        a_hat = self.struct_decoder(z, edge_index)
        return x_hat, a_hat


def gae_loss(X: torch.Tensor, X_hat: torch.Tensor,
             A: torch.Tensor, A_hat: torch.Tensor,
             alpha: float = 0.8) -> torch.Tensor:
    x_term = torch.norm(X - X_hat, p="fro") ** 2
    a_term = torch.norm(A - A_hat, p="fro") ** 2
    return alpha * x_term + (1.0 - alpha) * a_term


@torch.no_grad()
def reconstruction_scores_per_node(X: torch.Tensor, X_hat: torch.Tensor,
                                   A: torch.Tensor, A_hat: torch.Tensor,
                                   alpha: float = 0.8) -> torch.Tensor:
    x_err = torch.sum((X - X_hat) ** 2, dim=1)  # [N]
    a_err = torch.sum((A - A_hat) ** 2, dim=1)  # [N]
    return alpha * x_err + (1.0 - alpha) * a_err


def train_acm_gae(acm_path: str,
                  epochs: int = 50,
                  alpha: float = 0.8,
                  lr: float = 0.004,
                  seed: int = 42,
                  max_nodes: int = 1500,
                  cpu_threads: int | None = None):
    if cpu_threads is not None:
        torch.set_num_threads(int(cpu_threads))
        torch.set_num_interop_threads(1)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_csr, A_csr, y_raw_full, y_bin_full = load_acm_mat_sparse(acm_path)
    X_sub, A_sub, y_raw, y_bin, idx_map = induced_subgraph(
        X_csr, A_csr, y_raw_full, y_bin_full, max_nodes=max_nodes, seed=seed
    )

    edge_index, _ = from_scipy_sparse_matrix(A_sub)

    X = torch.from_numpy(X_sub.toarray().astype(np.float32)).to(device)  # [N,F]
    A = torch.from_numpy(A_sub.toarray().astype(np.float32)).to(device)  # [N,N]
    edge_index = edge_index.long().to(device)
    y_bin_t = torch.from_numpy(y_bin.astype(np.int64)).to(device)

    N, Fdim = X.shape
    print(f"Loaded ACM induced subgraph: N={N}, F={Fdim}, edges={edge_index.size(1)} (max_nodes={max_nodes})")
    uniq, cnt = np.unique(y_raw, return_counts=True)
    print(f"Original Label unique values (subset): {uniq.tolist()} (counts: {cnt.tolist()})")
    uniqb, cntb = torch.unique(y_bin_t.cpu(), return_counts=True)
    print(f"Binary labels for AUC (subset): {uniqb.tolist()} (counts: {cntb.tolist()})")

    model = GraphAutoencoder(in_dim=Fdim).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()                # required
        X_hat, A_hat = model(X, edge_index)  # required
        loss = gae_loss(X, X_hat, A, A_hat, alpha=alpha)  # required
        loss.backward()                      # required
        optimizer.step()                     # required

        if epoch % 5 == 0:
            model.eval()
            X_hat_eval, A_hat_eval = model(X, edge_index)
            scores = reconstruction_scores_per_node(X, X_hat_eval, A, A_hat_eval, alpha=alpha)

            y_true = y_bin_t.detach().cpu().numpy()
            y_scores = scores.detach().cpu().numpy()

            if len(np.unique(y_true)) < 2:
                print(f"Epoch {epoch:03d} | loss={loss.item():.6f} | ROC AUC=nan (single class)")
            else:
                auc = roc_auc_score(y_true, y_scores)
                print(f"Epoch {epoch:03d} | loss={loss.item():.6f} | ROC AUC={auc:.6f}")
        else:
            print(f"Epoch {epoch:03d} | loss={loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        X_hat, A_hat = model(X, edge_index)
        scores = reconstruction_scores_per_node(X, X_hat, A, A_hat, alpha=alpha)
        k = min(10, scores.numel())
        topk = torch.topk(scores, k=k).indices.cpu().numpy()

    print("\nTop anomalous nodes (by reconstruction score) — showing original node ids:")
    for rank, local_idx in enumerate(topk, start=1):
        orig_id = int(idx_map[local_idx])
        print(
            f"{rank:2d}. local_node={int(local_idx)}  orig_node={orig_id}  "
            f"score={float(scores[local_idx].cpu()):.6f}  label_bin={int(y_bin_t[local_idx].cpu())}"
        )

    return model


if __name__ == "__main__":
    ACM_PATH = "ACM.mat"
    if not os.path.exists(ACM_PATH):
        raise FileNotFoundError(
            f"Cannot find {ACM_PATH}. Place ACM.mat next to this script or set ACM_PATH to its location."
        )

    train_acm_gae(
        acm_path=ACM_PATH,
        epochs=50,
        alpha=0.8,
        lr=0.004,
        seed=42,
        max_nodes=1500,
        cpu_threads=4
    )
