import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


def principal_eigenvalue_weighted_adjacency(ego: nx.Graph) -> float:
    if ego.number_of_nodes() <= 1 or ego.number_of_edges() == 0:
        return 0.0
    A = nx.to_numpy_array(ego, weight="weight", dtype=float)
    eigvals = np.linalg.eigvalsh(A)
    return float(np.max(eigvals))


def compute_egonet_features_basic(G: nx.Graph, use_weights: bool) -> None:
    N_attr, E_attr, W_attr, L_attr = {}, {}, {}, {}

    for i in G.nodes():
        ego = nx.ego_graph(G, i, radius=1, center=True)
        N_i = len(list(G.neighbors(i)))
        E_i = ego.number_of_edges()

        if use_weights:
            W_i = sum(d.get("weight", 1.0) for _, _, d in ego.edges(data=True))
            lam = principal_eigenvalue_weighted_adjacency(ego)
        else:
            W_i = float(E_i)
            lam = 0.0

        N_attr[i] = int(N_i)
        E_attr[i] = int(E_i)
        W_attr[i] = float(W_i)
        L_attr[i] = float(lam)

    nx.set_node_attributes(G, N_attr, "N_i")
    nx.set_node_attributes(G, E_attr, "E_i")
    nx.set_node_attributes(G, W_attr, "W_i")
    nx.set_node_attributes(G, L_attr, "lambda_w")


def fit_power_law(G: nx.Graph, x_attr: str, y_attr: str):
    xs, ys = [], []
    for i in G.nodes():
        x = float(G.nodes[i].get(x_attr, 0.0))
        y = float(G.nodes[i].get(y_attr, 0.0))
        if x > 0 and y > 0:
            xs.append(x)
            ys.append(y)

    if len(xs) < 2:
        return 1.0, 1.0

    X = np.log(np.array(xs, dtype=float)).reshape(-1, 1)
    y = np.log(np.array(ys, dtype=float))

    model = LinearRegression()
    model.fit(X, y)

    theta = float(model.coef_[0])
    C = float(np.exp(model.intercept_))
    return C, theta


def oddball_score(y_i: float, y_hat: float) -> float:
    if y_i <= 0 or y_hat <= 0:
        return 0.0
    ratio = max(y_i, y_hat) / min(y_i, y_hat)
    return float(ratio * np.log(abs(y_i - y_hat) + 1.0))


def compute_scores(G: nx.Graph, x_attr: str, y_attr: str, score_attr: str) -> None:
    C, theta = fit_power_law(G, x_attr=x_attr, y_attr=y_attr)

    scores = {}
    for i in G.nodes():
        x = float(G.nodes[i].get(x_attr, 0.0))
        y = float(G.nodes[i].get(y_attr, 0.0))
        y_hat = C * (x ** theta) if x > 0 else 0.0
        scores[i] = oddball_score(y, y_hat)

    nx.set_node_attributes(G, scores, score_attr)
    return C, theta


def make_connected_by_random_edges(G: nx.Graph, rng: random.Random) -> None:
    while not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        c0 = list(comps[0])
        c1 = list(comps[1])
        u = rng.choice(c0)
        v = rng.choice(c1)
        G.add_edge(u, v)


def draw_highlight_topk(G: nx.Graph, score_attr: str, k: int, title: str) -> list:
    nodes_sorted = sorted(G.nodes(), key=lambda n: G.nodes[n].get(score_attr, 0.0), reverse=True)
    topk = nodes_sorted[:k]
    topk_set = set(topk)

    node_colors = ["red" if n in topk_set else "lightgray" for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 7))
    nx.draw(G, pos, node_color=node_colors, node_size=25, width=0.3, with_labels=False)
    plt.title(title)
    plt.show()

    print(f"Top {k} nodes by {score_attr}:")
    for rank, n in enumerate(topk, start=1):
        print(f"{rank:2d}. node={n}  {score_attr}={G.nodes[n].get(score_attr, 0.0):.6f}")

    return topk


def ex2_part1_cliques(seed: int = 42) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    G_reg = nx.random_regular_graph(d=3, n=100, seed=seed)

    G_cave = nx.connected_caveman_graph(l=10, k=20)

    G = nx.union(G_reg, G_cave, rename=("R", "C"))

    make_connected_by_random_edges(G, rng)

    plt.figure(figsize=(12, 7))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=25, width=0.3, with_labels=False)
    plt.title("Ex.2.1 — Merged graph (random regular + connected caveman), made connected")
    plt.show()

    compute_egonet_features_basic(G, use_weights=False)
    C, theta = compute_scores(G, x_attr="N_i", y_attr="E_i", score_attr="score_clique")
    print(f"[Ex2.1] Fitted power-law on (E_i vs N_i): E = C*N^theta with C={C:.6f}, theta={theta:.6f}")

    draw_highlight_topk(
        G,
        score_attr="score_clique",
        k=10,
        title="Ex.2.1 — Top 10 clique candidates (OddBall-style score on E_i vs N_i)"
    )


def ex2_part2_heavy_vicinity(seed: int = 42) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    G3 = nx.random_regular_graph(d=3, n=100, seed=seed)
    G5 = nx.random_regular_graph(d=5, n=100, seed=seed + 1)

    G = nx.union(G3, G5, rename=("G3_", "G5_"))

    for u, v in list(G.edges()):
        G[u][v]["weight"] = 1.0

    nodes = list(G.nodes())
    chosen = rng.sample(nodes, 2)
    print(f"[Ex2.2] Chosen heavy-vicinity seed nodes: {chosen}")

    for center in chosen:
        ego = nx.ego_graph(G, center, radius=1, center=True)
        for u, v in ego.edges():
            G[u][v]["weight"] += 10.0

    plt.figure(figsize=(12, 7))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=25, width=0.3, with_labels=False)
    plt.title("Ex.2.2 — Merged graph (deg3 + deg5) with heavy-vicinity weight boosts")
    plt.show()

    compute_egonet_features_basic(G, use_weights=True)
    C, theta = compute_scores(G, x_attr="E_i", y_attr="W_i", score_attr="score_heavy")
    print(f"[Ex2.2] Fitted power-law on (W_i vs E_i): W = C*E^theta with C={C:.6f}, theta={theta:.6f}")

    draw_highlight_topk(
        G,
        score_attr="score_heavy",
        k=4,
        title="Ex.2.2 — Top 4 HeavyVicinity candidates (OddBall-style score on W_i vs E_i)"
    )


def main():
    ex2_part1_cliques(seed=42)
    ex2_part2_heavy_vicinity(seed=42)


if __name__ == "__main__":
    main()
