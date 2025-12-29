import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor


def load_weighted_graph_from_edgelist(path: str, max_rows: int = 1500) -> nx.Graph:
    G = nx.Graph()
    edge_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if edge_count >= max_rows:
                break

            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            u = int(parts[0])
            v = int(parts[1])

            if G.has_edge(u, v):
                G[u][v]["weight"] += 1.0
            else:
                G.add_edge(u, v, weight=1.0)

            edge_count += 1

    return G


def principal_eigenvalue_weighted_adjacency(ego: nx.Graph) -> float:
    if ego.number_of_nodes() <= 1 or ego.number_of_edges() == 0:
        return 0.0

    A = nx.to_numpy_array(ego, weight="weight", dtype=float)
    eigvals = np.linalg.eigvalsh(A)  # symmetric -> real eigenvalues
    return float(np.max(eigvals))


def compute_egonet_features(G: nx.Graph) -> None:
    N_attr = {}
    E_attr = {}
    W_attr = {}
    L_attr = {}

    for i in G.nodes():
        neighbors = list(G.neighbors(i))
        N_i = len(neighbors)

        ego = nx.ego_graph(G, i, radius=1, center=True)  # node + neighbors

        E_i = ego.number_of_edges()
        W_i = sum(d.get("weight", 1.0) for _, _, d in ego.edges(data=True))
        lam = principal_eigenvalue_weighted_adjacency(ego)

        N_attr[i] = N_i
        E_attr[i] = E_i
        W_attr[i] = float(W_i)
        L_attr[i] = lam

    nx.set_node_attributes(G, N_attr, "N_i")
    nx.set_node_attributes(G, E_attr, "E_i")
    nx.set_node_attributes(G, W_attr, "W_i")
    nx.set_node_attributes(G, L_attr, "lambda_w")


def fit_power_law_linear_regression(G: nx.Graph):
    Ns = []
    Es = []
    for i in G.nodes():
        N_i = G.nodes[i].get("N_i", 0)
        E_i = G.nodes[i].get("E_i", 0)
        if N_i > 0 and E_i > 0:
            Ns.append(N_i)
            Es.append(E_i)

    if len(Ns) < 2:
        return 1.0, 1.0

    X = np.log(np.array(Ns, dtype=float)).reshape(-1, 1)
    y = np.log(np.array(Es, dtype=float))

    model = LinearRegression()
    model.fit(X, y)

    theta = float(model.coef_[0])
    C = float(np.exp(model.intercept_))
    return C, theta


def oddball_score(E_i: float, E_hat: float) -> float:
    if E_i <= 0 or E_hat <= 0:
        return 0.0

    ratio = max(E_i, E_hat) / min(E_i, E_hat)
    return float(ratio * np.log(abs(E_i - E_hat) + 1.0))


def compute_oddball_scores(G: nx.Graph, C: float, theta: float) -> None:
    scores = {}
    for i in G.nodes():
        N_i = float(G.nodes[i].get("N_i", 0))
        E_i = float(G.nodes[i].get("E_i", 0))

        E_hat = C * (N_i ** theta) if N_i > 0 else 0.0
        scores[i] = oddball_score(E_i, E_hat)

    nx.set_node_attributes(G, scores, "score_oddball")


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if np.isclose(vmax - vmin, 0.0):
        return np.zeros_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def compute_lof_scores(G: nx.Graph, n_neighbors: int = 50) -> None:
    nodes = list(G.nodes())
    X = np.array([[G.nodes[i].get("E_i", 0), G.nodes[i].get("N_i", 0)] for i in nodes], dtype=float)

    if len(nodes) < 3:
        nx.set_node_attributes(G, {i: 0.0 for i in nodes}, "score_lof")
        return

    k = min(n_neighbors, len(nodes) - 1)
    lof = LocalOutlierFactor(n_neighbors=k)
    lof.fit_predict(X)

    raw = -lof.negative_outlier_factor_  # bigger -> more outlier
    nx.set_node_attributes(G, {node: float(raw[idx]) for idx, node in enumerate(nodes)}, "score_lof")


def compute_combined_scores(G: nx.Graph) -> None:
    nodes = list(G.nodes())
    odd = np.array([G.nodes[i].get("score_oddball", 0.0) for i in nodes], dtype=float)
    lof = np.array([G.nodes[i].get("score_lof", 0.0) for i in nodes], dtype=float)

    odd_n = minmax_normalize(odd)
    combined = odd_n + lof

    nx.set_node_attributes(G, {nodes[idx]: float(combined[idx]) for idx in range(len(nodes))}, "score_combined")


def draw_top10(G: nx.Graph, score_attr: str, title: str) -> None:
    nodes_sorted = sorted(G.nodes(), key=lambda i: G.nodes[i].get(score_attr, 0.0), reverse=True)
    top10 = set(nodes_sorted[:10])

    node_colors = ["red" if i in top10 else "lightgray" for i in G.nodes()]

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=30, node_color=node_colors, with_labels=False, width=0.3)
    plt.title(title)
    plt.show()

    print(f"Top 10 nodes by {score_attr}:")
    for rank, node in enumerate(nodes_sorted[:10], start=1):
        print(f"{rank:2d}. node={node}  {score_attr}={G.nodes[node].get(score_attr, 0.0):.6f}")


def main():
    path = "ca-AstroPh.txt"
    G = load_weighted_graph_from_edgelist(path, max_rows=1500)

    compute_egonet_features(G)

    C, theta = fit_power_law_linear_regression(G)
    print(f"Fitted power-law: E = C * N^theta with C={C:.6f}, theta={theta:.6f}")

    compute_oddball_scores(G, C, theta)
    draw_top10(G, "score_oddball", "Exercise 1 (OddBall) — Top 10 anomalies by OddBall score")

    compute_lof_scores(G, n_neighbors=50)
    compute_combined_scores(G)
    draw_top10(G, "score_combined", "Exercise 1 (OddBall + LOF) — Top 10 anomalies by combined score")


if __name__ == "__main__":
    main()
