import numpy as np
from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD

data = loadmat("shuttle 1.mat")
X = data["X"]
y = data["y"].ravel().astype(int)   

print("Data shape:", X.shape)
print("Label distribution (0=inlier, 1=outlier):", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,   
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

contamination = y_train.mean()
print(f"Train contamination: {contamination:.4f}")

def evaluate_model(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)
    scores = model.decision_function(X_test)

    ba = balanced_accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, scores)

    print(f"\n{name}")
    print(f"  Balanced accuracy (test): {ba:.4f}")
    print(f"  ROC AUC (test):           {roc:.4f}")

    return ba, roc

ocsvm = OCSVM(
    kernel="rbf",
    gamma="scale",
    contamination=contamination
)
ocsvm.fit(X_train_s)

evaluate_model(ocsvm, X_test_s, y_test, name="OCSVM (RBF)")

architectures = {
    "DeepSVDD small [32, 16]":          [32, 16],
    "DeepSVDD medium [64, 32]":         [64, 32],
    "DeepSVDD large [128, 64, 32]":     [128, 64, 32],
}

results = {}

for arch_name, hidden in architectures.items():
    print(f"\nTraining {arch_name} ...")

    deep = DeepSVDD(
        n_features=X_train_s.shape[1],
        hidden_neurons=hidden,
        epochs=50,
        batch_size=64,
        contamination=contamination,
        random_state=42,
        verbose=0,
    )
    deep.fit(X_train_s)

    ba, roc = evaluate_model(deep, X_test_s, y_test, name=arch_name)
    results[arch_name] = (ba, roc)

print("\n=== Summary of DeepSVDD architectures (TEST) ===")
for arch_name, (ba, roc) in results.items():
    print(f"{arch_name}: BA={ba:.4f}, ROC AUC={roc:.4f}")
