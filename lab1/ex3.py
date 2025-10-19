import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

RANDOM_STATE = 42
CONTAM = 0.10
N_TRAIN = 1000

X_train, X_test, y_train, y_test = generate_data(
    n_train=N_TRAIN,
    n_test=0,            
    n_features=1,       
    contamination=CONTAM,
    random_state=RANDOM_STATE
)

x = X_train.ravel()
mu = np.mean(x)
sigma = np.std(x, ddof=0)  
z_abs = np.abs((x - mu) / (sigma if sigma > 0 else 1.0))

thr = np.quantile(z_abs, 1.0 - CONTAM)

y_pred = (z_abs >= thr).astype(int)

tn, fp, fn, tp = confusion_matrix(y_train, y_pred, labels=[0, 1]).ravel()
ba = balanced_accuracy_score(y_train, y_pred)

print(f"Mean={mu:.4f}, Std={sigma:.4f}, Threshold(|Z|)={thr:.4f}")
print(f"Counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Balanced Accuracy: {ba:.4f}")
