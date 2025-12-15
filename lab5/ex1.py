import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
mean = [5, 10, 2]
cov = [[3, 2, 2], [2, 10, 1], [2, 1, 2]]
data = np.random.multivariate_normal(mean, cov, size=500)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', alpha=0.6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Original 3D Dataset')
plt.show()

data_centered = data - np.mean(data, axis=0)

cov_matrix = (data_centered.T @ data_centered) / len(data)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Eigenvalues (sorted descending): {eigenvalues}")

fig, ax = plt.subplots(figsize=(10, 6))

cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
ax.step(range(1, len(eigenvalues) + 1), cumulative_variance, where='mid', 
        label='Cumulative Explained Variance', color='red', linewidth=2)

ax2 = ax.twinx()
individual_variance = eigenvalues / np.sum(eigenvalues)
ax2.bar(range(1, len(eigenvalues) + 1), individual_variance, 
        alpha=0.5, label='Individual Variance', color='blue')

ax.set_xlabel('Principal Component')
ax.set_ylabel('Cumulative Explained Variance', color='red')
ax2.set_ylabel('Individual Variance', color='blue')
ax.set_title('Explained Variance Analysis')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()

data_projected = data_centered @ eigenvectors

contamination = 0.1

pc3_values = data_projected[:, 2]
pc3_mean = np.mean(pc3_values)
pc3_deviations = np.abs(pc3_values - pc3_mean)

threshold_pc3 = np.quantile(pc3_deviations, 1 - contamination)
outliers_pc3 = pc3_deviations > threshold_pc3

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[~outliers_pc3, 0], data[~outliers_pc3, 1], data[~outliers_pc3, 2], 
           c='blue', alpha=0.6, label='Normal')
ax.scatter(data[outliers_pc3, 0], data[outliers_pc3, 1], data[outliers_pc3, 2], 
           c='red', alpha=0.8, label='Anomaly', s=100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Outlier Detection based on 3rd Principal Component')
ax.legend()
plt.show()

print(f"Number of outliers (PC3): {np.sum(outliers_pc3)}")

pc2_values = data_projected[:, 1]
pc2_mean = np.mean(pc2_values)
pc2_deviations = np.abs(pc2_values - pc2_mean)

threshold_pc2 = np.quantile(pc2_deviations, 1 - contamination)
outliers_pc2 = pc2_deviations > threshold_pc2

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[~outliers_pc2, 0], data[~outliers_pc2, 1], data[~outliers_pc2, 2], 
           c='blue', alpha=0.6, label='Normal')
ax.scatter(data[outliers_pc2, 0], data[outliers_pc2, 1], data[outliers_pc2, 2], 
           c='red', alpha=0.8, label='Anomaly', s=100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Outlier Detection based on 2nd Principal Component')
ax.legend()
plt.show()

print(f"Number of outliers (PC2): {np.sum(outliers_pc2)}")

data_transformed = data_centered @ eigenvectors

data_normalized = data_transformed / np.sqrt(eigenvalues)

anomaly_scores = np.sum(data_normalized ** 2, axis=1)

threshold_all = np.quantile(anomaly_scores, 1 - contamination)
outliers_all = anomaly_scores > threshold_all

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[~outliers_all, 0], data[~outliers_all, 1], data[~outliers_all, 2], 
           c='blue', alpha=0.6, label='Normal')
ax.scatter(data[outliers_all, 0], data[outliers_all, 1], data[outliers_all, 2], 
           c='red', alpha=0.8, label='Anomaly', s=100)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Outlier Detection based on Normalized Distance (All PCs)')
ax.legend()
plt.show()

print(f"Number of outliers (All PCs): {np.sum(outliers_all)}")