import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD

X_train, X_test, y_train, y_test = generate_data(
    n_train=300,
    n_test=200,
    n_features=3,
    contamination=0.15,
    random_state=42
)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("Shapes:", X_train.shape, X_test.shape)
print("Class distribution train (0=inlier,1=outlier):",
      np.bincount(y_train))
print("Class distribution test:",
      np.bincount(y_test))

def evaluate_model(model, X_test, y_test, name="model"):
    """
    Computes balanced accuracy + ROC AUC on test set.
    y_test is in PyOD format: 0 = inlier, 1 = outlier.
    """
    y_pred = model.predict(X_test)
    scores_test = model.decision_function(X_test)

    ba = balanced_accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, scores_test)

    print(f"{name}:")
    print(f"  Balanced accuracy = {ba:.4f}")
    print(f"  ROC AUC          = {roc:.4f}\n")
    return y_pred, scores_test

def plot_3d(ax, X, labels, title):
    """
    X: (n_samples, 3)
    labels: 0=inlier, 1=outlier
    """
    inliers = X[labels == 0]
    outliers = X[labels == 1]

    ax.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2],
               s=15, alpha=0.7, label="inliers")
    ax.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2],
               s=15, alpha=0.7, marker='^', label="outliers")

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.legend(loc="best")

ocsvm_linear = OCSVM(kernel="linear",
                     contamination=0.15)

ocsvm_linear.fit(X_train)

y_pred_linear_test, _ = evaluate_model(
    ocsvm_linear, X_test, y_test, name="OCSVM (linear kernel)"
)
y_pred_linear_train = ocsvm_linear.predict(X_train)

fig = plt.figure(figsize=(14, 10))

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
plot_3d(ax1, X_train, y_train, "TRAIN – ground truth")

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
plot_3d(ax2, X_test, y_test, "TEST – ground truth")

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
plot_3d(ax3, X_train, y_pred_linear_train, "TRAIN – OCSVM linear (predicted)")

ax4 = fig.add_subplot(2, 2, 4, projection='3d')
plot_3d(ax4, X_test, y_pred_linear_test, "TEST – OCSVM linear (predicted)")

plt.suptitle("OCSVM with linear kernel")
plt.tight_layout()
plt.show()

ocsvm_rbf = OCSVM(kernel="rbf",
                  contamination=0.15,
                  gamma='scale') 

ocsvm_rbf.fit(X_train)

y_pred_rbf_test, _ = evaluate_model(
    ocsvm_rbf, X_test, y_test, name="OCSVM (RBF kernel)"
)

deep_svdd = DeepSVDD(
    n_features=X_train.shape[1],
    hidden_neurons=[64, 32], 
    epochs=50,
    batch_size=32,
    contamination=0.15,
    random_state=42,
    verbose=1
)

deep_svdd.fit(X_train)

y_pred_deep_test, _ = evaluate_model(
    deep_svdd, X_test, y_test, name="DeepSVDD"
)
y_pred_deep_train = deep_svdd.predict(X_train)

fig2 = plt.figure(figsize=(14, 10))

ax1 = fig2.add_subplot(2, 2, 1, projection='3d')
plot_3d(ax1, X_train, y_train, "TRAIN – ground truth")

ax2 = fig2.add_subplot(2, 2, 2, projection='3d')
plot_3d(ax2, X_test, y_test, "TEST – ground truth")

ax3 = fig2.add_subplot(2, 2, 3, projection='3d')
plot_3d(ax3, X_train, y_pred_deep_train, "TRAIN – DeepSVDD (predicted)")

ax4 = fig2.add_subplot(2, 2, 4, projection='3d')
plot_3d(ax4, X_test, y_pred_deep_test, "TEST – DeepSVDD (predicted)")

plt.suptitle("DeepSVDD")
plt.tight_layout()
plt.show()