import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

mat = loadmat("cardio.mat")
X = mat["X"]
y = mat["y"].ravel().astype(int)

contamination = float(np.mean(y))
print(f"Dataset shape: X={X.shape}, contamination={contamination:.3f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_family = "knn"
n_list = np.linspace(30, 120, 10, dtype=int)

models = []
train_scores_raw = []
test_scores_raw = []
per_model_metrics = []

for n in n_list:
    if model_family == "knn":
        clf = KNN(n_neighbors=int(n), contamination=contamination)
    else:
        clf = LOF(n_neighbors=int(n), contamination=contamination)

    clf.fit(X_train)

    s_train = clf.decision_scores_
    s_test = clf.decision_function(X_test)

    y_pred_tr = (s_train > clf.threshold_).astype(int)
    y_pred_te = (s_test > clf.threshold_).astype(int)

    bacc_tr = balanced_accuracy_score(y_train, y_pred_tr)
    bacc_te = balanced_accuracy_score(y_test,  y_pred_te)
    per_model_metrics.append((int(n), bacc_tr, bacc_te))

    models.append(clf)
    train_scores_raw.append(s_train)
    test_scores_raw.append(s_test)

print("\nPer-model balanced accuracy (train | test):")
for n, bt, be in per_model_metrics:
    print(f"n_neighbors={n:>3}  ->  BA train={bt:.3f} | BA test={be:.3f}")

train_scores_raw = np.column_stack(train_scores_raw)
test_scores_raw  = np.column_stack(test_scores_raw)

train_scores_std, test_scores_std = standardizer(train_scores_raw, test_scores_raw)

comb_train_avg = average(train_scores_std)
comb_test_avg  = average(test_scores_std)

comb_train_max = maximization(train_scores_std)
comb_test_max  = maximization(test_scores_std)

q = 1.0 - contamination

thr_avg = np.quantile(comb_train_avg, q)
thr_max = np.quantile(comb_train_max, q)

y_pred_train_avg = (comb_train_avg > thr_avg).astype(int)
y_pred_test_avg  = (comb_test_avg  > thr_avg).astype(int)

y_pred_train_max = (comb_train_max > thr_max).astype(int)
y_pred_test_max  = (comb_test_max  > thr_max).astype(int)

ba_train_avg = balanced_accuracy_score(y_train, y_pred_train_avg)
ba_test_avg  = balanced_accuracy_score(y_test,  y_pred_test_avg)

ba_train_max = balanced_accuracy_score(y_train, y_pred_train_max)
ba_test_max  = balanced_accuracy_score(y_test,  y_pred_test_max)

print("\nEnsemble results (threshold from train quantile):")
print(f"[AVERAGE]      BA train={ba_train_avg:.3f} | BA test={ba_test_avg:.3f}")
print(f"[MAXIMIZATION] BA train={ba_train_max:.3f} | BA test={ba_test_max:.3f}")