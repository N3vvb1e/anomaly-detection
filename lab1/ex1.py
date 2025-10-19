from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
import numpy as np

X_train, X_test, y_train, y_test = generate_data(
    n_train=400, n_test=100, n_features=2, contamination=0.1, random_state=42
)

in_mask  = (y_train == 0)
out_mask = (y_train == 1)

plt.figure(figsize=(6, 5))
plt.scatter(X_train[in_mask, 0],  X_train[in_mask, 1],  s=18, label="Inliers (0)")
plt.scatter(X_train[out_mask, 0], X_train[out_mask, 1], s=24, marker="x", label="Outliers (1)")
plt.title("Training data (2D) — PyOD generate_data, contamination=0.1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()

# sanity check
assert X_train.ndim == 2 and X_test.ndim == 2
assert y_train.ndim == 1 and y_test.ndim == 1
assert X_train.shape[0] == y_train.shape[0]
assert X_test.shape[0]  == y_test.shape[0]
print("Shapes OK:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

