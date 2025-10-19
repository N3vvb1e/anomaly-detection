from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

RANDOM_STATE = 42
DATA_CONTAM = 0.1

X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=DATA_CONTAM,
    random_state=RANDOM_STATE
)

knn = KNN(contamination=DATA_CONTAM)
knn.fit(X_train)

def eval_split(name, X, y, clf):
    y_pred = clf.predict(X)
    scores = clf.decision_function(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    ba = balanced_accuracy_score(y, y_pred)

    print(f"\n=== {name} ===")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Balanced Accuracy: {ba:.4f}")

    return y_pred, scores, (tn, fp, fn, tp), ba

y_pred_tr, scores_tr, cm_tr, ba_tr = eval_split("TRAIN", X_train, y_train, knn)
y_pred_te, scores_te, cm_te, ba_te = eval_split("TEST",  X_test,  y_test,  knn)

fpr, tpr, thresholds = roc_curve(y_test, scores_te, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"KNN (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve — Test set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

sweep = [0.01, 0.05, 0.10, 0.20, 0.30]
print("\n=== Contamination sweep (model only) on TEST set ===")
print("contam\tTN\tFP\tFN\tTP\tBA")

for c in sweep:
    model = KNN(contamination=c)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    ba = balanced_accuracy_score(y_test, y_pred)
    print(f"{c:.2f}\t{tn}\t{fp}\t{fn}\t{tp}\t{ba:.4f}")
