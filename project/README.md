# ECOD Experiment Replication

Partial replication of experiments from **"ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions"**.

---

## 📁 Files

### Python Scripts

**`experiment2_interpretation.py`** - Replicates Experiment 2 (Interpretability)
- Analyzes Sample 70 from BreastW dataset using ECOD
- Computes dimensional outlier scores via ECDF tail probabilities
- Generates dimensional outlier graph showing which features contribute to anomaly detection
- Output: `dimensional_outlier_graph.png`

**`experiment3_runtime.py`** - Replicates Experiment 3 (Runtime Performance)
- Benchmarks ECOD vs. IForest, KNN, and LOF
- Tests scalability with varying sample sizes (n) and dimensions (d)
- Outputs: `runtime_scalability.png`, `runtime_n.csv`, `runtime_d.csv`

### Data Files

**`4_breastw.npz`** - Breast Cancer Wisconsin dataset (features `X`, labels `y`)

**`runtime_n.csv`** - Runtime results for varying sample sizes

**`runtime_d.csv`** - Runtime results for varying dimensions

---

## 🚀 Usage

```bash
# Install dependencies
pip install numpy pandas matplotlib scipy pyod

# Run experiments
python experiment2_interpretation.py
python experiment3_runtime.py
```

---

## 📖 Reference

```
Li, Z., Zhao, Y., Botta, N., Ionescu, C., & Hu, X. (2022).
ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions.
IEEE Transactions on Knowledge and Data Engineering.
```