import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyod.models.ecod import ECOD
from scipy.stats import skew

# Load the BreastW dataset
try:
    data = np.load('4_breastw.npz')
    X = data['X']
    y = data['y']
except FileNotFoundError:
    print("Error: '4_breastw.npz' not found. Please upload the file.")

# Train ECOD
clf = ECOD()
clf.fit(X)

# Select Sample 70 (Index 69 in 0-based Python indexing)
sample_idx = 69 
sample_data = X[sample_idx].reshape(1, -1)

# Verify if it is actually an outlier in Ground Truth
print(f"Sample {sample_idx+1} Ground Truth: {'Outlier' if y[sample_idx]==1 else 'Normal'}")

# Calculate Dimensional Outlier Scores

# Get skewness of the training data
feature_skewness = skew(X, axis=0)


# Calculate -log(P) score per dimension using tail probabilities
U_l_sample = np.zeros(X.shape[1])
U_r_sample = np.zeros(X.shape[1])
dim_scores = []
percentile_99_scores = []

for j in range(X.shape[1]):
    # If skew < 0, use left tail. If skew >= 0, use right tail (ECOD equation 6)
    # Calculate ECDF for sample 70
    col_data = X[:, j]
    val = sample_data[0, j]
    
    # Left tail prob: portion of data <= val
    p_left = (np.sum(col_data <= val) + 1) / (len(col_data) + 1) # +1 for smoothing
    # Right tail prob: portion of data >= val
    p_right = (np.sum(col_data >= val) + 1) / (len(col_data) + 1)
    
    # Select tail based on skewness
    if feature_skewness[j] < 0:
        p_selected = p_left
    else:
        p_selected = p_right
        
    # Dimensional Outlier Score = -log(P_tail)
    score = -np.log(p_selected)
    dim_scores.append(score)
    
    # Calculate 99th percentile threshold for this dimension
    if feature_skewness[j] < 0:
        all_probs = (np.array([np.sum(col_data <= v) for v in col_data]) + 1) / (len(col_data) + 1)
    else:
        all_probs = (np.array([np.sum(col_data >= v) for v in col_data]) + 1) / (len(col_data) + 1)
        
    all_scores = -np.log(all_probs)
    threshold = np.percentile(all_scores, 99)
    percentile_99_scores.append(threshold)

# Plotting (Replicating Figure 2 style)
dims = range(1, X.shape[1] + 1)

plt.figure(figsize=(10, 4))
plt.plot(dims, dim_scores, 'k.-', label='Dimensional Outlier Score', linewidth=1.5)
plt.plot(dims, percentile_99_scores, 'g-.', label='99% Percentile Line', linewidth=1.0)

# Highlight points crossing the threshold
for i, (s, t) in enumerate(zip(dim_scores, percentile_99_scores)):
    if s > t:
        plt.scatter(i+1, s, s=150, facecolors='none', edgecolors='r', linestyle='--', linewidth=1.5)

plt.xlabel('Dimension #')
plt.ylabel('Dimensional Outlier Score')
plt.title(f'Dimensional Outlier Graph for Sample {sample_idx+1} (BreastW)')
plt.xticks(dims)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


plt.savefig('dimensional_outlier_graph.png', dpi=300)
print("Graph saved as 'dimensional_outlier_graph.png'")
plt.show()