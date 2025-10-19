import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

d = 3           # dimensionality (>=2 so it's truly multi-d)
N  = 1000             
CONTAM = 0.10               

n_out = int(N * CONTAM)
n_in  = N - n_out

mu_in = np.array([1.0, -2.0, 0.5])
# make Σ positive-definite with correlations
A = np.array([[1.0, 0.4, 0.2],
              [0.4, 1.2, 0.3],
              [0.2, 0.3, 0.8]])
Sigma_in = A @ A.T
L_in = np.linalg.cholesky(Sigma_in)     # Σ = L L^T

X_in = rng.standard_normal(size=(n_in, d))
Y_in = (X_in @ L_in.T) + mu_in

mu_out = mu_in + np.array([3.0, -3.0, 2.0])     # shift
Sigma_out = 2.5 * Sigma_in                      # inflate covariance
L_out = np.linalg.cholesky(Sigma_out)
X_out = rng.standard_normal(size=(n_out, d))
Y_out = (X_out @ L_out.T) + mu_out

Y = np.vstack([Y_in, Y_out])
y_true = np.hstack([np.zeros(n_in, dtype=int), np.ones(n_out, dtype=int)])

# z = L^{-1}(y - μ)
Z = np.linalg.solve(L_in, (Y - mu_in).T).T      # shape (N, d)
scores = np.linalg.norm(Z, axis=1)              # ||z||_2 as anomaly score

thr = np.quantile(scores, 1.0 - CONTAM)
y_pred = (scores >= thr).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
ba = balanced_accuracy_score(y_true, y_pred)

print(f"Effective true contamination: {y_true.mean():.3f}")
print(f"Predicted anomaly rate     : {y_pred.mean():.3f}")
print(f"Threshold on ||z||          : {thr:.3f}")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Balanced Accuracy           : {ba:.4f}")
