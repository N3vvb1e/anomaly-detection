import numpy as np
from scipy.io import loadmat

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.metrics import balanced_accuracy_score, make_scorer

data = loadmat("cardio 1.mat") 
X = data["X"]
y_pyod = data["y"].ravel().astype(int) 

print("Data shape:", X.shape)
print("Label distribution (pyod format, 0=inlier,1=outlier):",
      np.bincount(y_pyod))

y_skl = -2 * y_pyod + 1
print("Unique labels after conversion to sklearn format:", np.unique(y_skl))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_skl,
    train_size=0.4,  
    random_state=42,
    stratify=y_skl
)

print("Train shape:", X_train.shape, " Test shape:", X_test.shape)

cont_train = (y_train == -1).mean()
print(f"Train contamination (fraction of outliers): {cont_train:.4f}")

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ocsvm", OneClassSVM())
])

nu_candidates = [
    0.01,
    max(cont_train / 2, 0.001),
    cont_train,
    min(cont_train * 2, 0.49)
]

param_grid = [
    {
        "ocsvm__kernel": ["linear"],
        "ocsvm__nu": nu_candidates
    },
    {
        "ocsvm__kernel": ["rbf", "poly", "sigmoid"],
        "ocsvm__gamma": ["scale", 0.01, 0.1, 1.0],
        "ocsvm__nu": nu_candidates
    }
]

scorer = make_scorer(balanced_accuracy_score)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scorer,
    cv=5, 
    n_jobs=-1,
    verbose=1,
    refit=True   
)

print("\nRunning GridSearchCV...")
grid.fit(X_train, y_train)

print("\nBest CV balanced accuracy: {:.4f}".format(grid.best_score_))
print("Best parameters found:")
for k, v in grid.best_params_.items():
    print(f"  {k}: {v}")

best_model = grid.best_estimator_

y_pred_test = best_model.predict(X_test)  
ba_test = balanced_accuracy_score(y_test, y_pred_test)

print("\nBalanced accuracy on TEST set with best params: {:.4f}".format(ba_test))

y_pred_train = best_model.predict(X_train)
ba_train = balanced_accuracy_score(y_train, y_pred_train)
print("Balanced accuracy on TRAIN set (with best params): {:.4f}".format(ba_train))
