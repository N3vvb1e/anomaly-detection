import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyod.models.ecod import ECOD
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.utils.data import generate_data

n_range = [1000, 5000, 10000, 20000] 
fixed_d = 20

d_range = [10, 50, 100, 200, 500]
fixed_n = 5000

def get_fresh_models():
    """Return a fresh dictionary of models to ensure no state is carried over."""
    return {
        'ECOD': ECOD(),
        'IForest': IForest(random_state=42, n_jobs=1),
        'KNN': KNN(n_jobs=1),
        'LOF': LOF(n_jobs=1)
    }

def measure_runtime(model, X):
    start = time.time()
    model.fit(X)
    model.decision_function(X) 
    return time.time() - start

print("Running Experiment A: Varying Sample Size (n)...")
results_n = {'n_samples': n_range, 'ECOD': [], 'IForest': [], 'KNN': [], 'LOF': []}

for n in n_range:
    print(f"  Testing n={n}...")
    X, _, _, _ = generate_data(n_train=n, n_test=0, n_features=fixed_d, contamination=0.1)
    models = get_fresh_models()
    
    for name, clf in models.items():
        duration = measure_runtime(clf, X)
        results_n[name].append(duration)


print("\n" + "="*40)
print(f"RESULTS: Runtime (seconds) vs Sample Size (d={fixed_d})")
print("="*40)
df_n = pd.DataFrame(results_n)
print(df_n)
print("="*40 + "\n")


print("Running Experiment B: Varying Dimensions (d)...")
results_d = {'n_features': d_range, 'ECOD': [], 'IForest': [], 'KNN': [], 'LOF': []}

for d in d_range:
    print(f"  Testing d={d}...")
    X, _, _, _ = generate_data(n_train=fixed_n, n_test=0, n_features=d, contamination=0.1)
    
    models = get_fresh_models()
    
    for name, clf in models.items():
        duration = measure_runtime(clf, X)
        results_d[name].append(duration)


print("\n" + "="*40)
print(f"RESULTS: Runtime (seconds) vs Dimensions (n={fixed_n})")
print("="*40)
df_d = pd.DataFrame(results_d)
print(df_d)
print("="*40 + "\n")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for name in ['ECOD', 'IForest', 'KNN', 'LOF']:
    ax1.plot(results_n['n_samples'], results_n[name], marker='o', label=name)
ax1.set_title(f'Runtime vs Sample Size (fixed d={fixed_d})')
ax1.set_xlabel('Number of Samples (n)')
ax1.set_ylabel('Time (seconds)')
ax1.set_yscale('log') 
ax1.legend()
ax1.grid(True, which="both", ls="-", alpha=0.5)

for name in ['ECOD', 'IForest', 'KNN', 'LOF']:
    ax2.plot(results_d['n_features'], results_d[name], marker='o', label=name)
ax2.set_title(f'Runtime vs Dimensions (fixed n={fixed_n})')
ax2.set_xlabel('Number of Dimensions (d)')
ax2.set_ylabel('Time (seconds)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, which="both", ls="-", alpha=0.5)

plt.tight_layout()
plt.savefig('runtime_scalability.png', dpi=300)
print("Success! Plot saved as 'runtime_scalability.png'")
plt.show()

df_n.to_csv('runtime_n.csv', index=False)
df_d.to_csv('runtime_d.csv', index=False)