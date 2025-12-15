import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from pyod.utils.utility import standardizer

data = loadmat('shuttle.mat')
X = data['X']
y = data['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

X_train_norm, X_test_norm = standardizer(X_train, X_test)

contamination_rate = np.sum(y_train == 1) / len(y_train)
print(f"Training Contamination Rate: {contamination_rate:.4f}")


print("\n--- Fitting PCA ---")
clf_pca = PCA(contamination=contamination_rate)
clf_pca.fit(X_train_norm)

y_train_pred_pca = clf_pca.labels_ 
y_test_pred_pca = clf_pca.predict(X_test_norm)

explained_variance_ratio = clf_pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))

plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 
        alpha=0.5, align='center', label='Individual explained variance')

plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, 
         where='mid', label='Cumulative explained variance')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('PCA: Explained Variance')
plt.tight_layout()
plt.show()

b_acc_train_pca = balanced_accuracy_score(y_train, y_train_pred_pca)
b_acc_test_pca = balanced_accuracy_score(y_test, y_test_pred_pca)

print(f"PCA Balanced Accuracy (Train): {b_acc_train_pca:.4f}")
print(f"PCA Balanced Accuracy (Test):  {b_acc_test_pca:.4f}")


print("\n--- Fitting KPCA ---")

max_kpca_samples = 2000 

if len(X_train_norm) > max_kpca_samples:
    print(f"Subsampling training data to {max_kpca_samples} samples for KPCA fitting...")
    idx = np.random.choice(np.arange(len(X_train_norm)), max_kpca_samples, replace=False)
    X_train_kpca_fit = X_train_norm[idx]
else:
    X_train_kpca_fit = X_train_norm

clf_kpca = KPCA(contamination=contamination_rate)
clf_kpca.fit(X_train_kpca_fit)

print("Predicting KPCA labels (this may take a moment)...")

y_train_pred_kpca = clf_kpca.predict(X_train_norm)
y_test_pred_kpca = clf_kpca.predict(X_test_norm)

b_acc_train_kpca = balanced_accuracy_score(y_train, y_train_pred_kpca)
b_acc_test_kpca = balanced_accuracy_score(y_test, y_test_pred_kpca)

print(f"KPCA Balanced Accuracy (Train): {b_acc_train_kpca:.4f}")
print(f"KPCA Balanced Accuracy (Test):  {b_acc_test_kpca:.4f}")