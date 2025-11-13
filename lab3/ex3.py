import warnings
warnings.filterwarnings("ignore", message="'pin_memory' argument is set as true")

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.dif import DIF

def load_shuttle_mat(path="shuttle.mat"):
    mat = loadmat(path)
    X = mat.get("X")
    y = mat.get("y")
    if X is None or y is None:
        raise KeyError(f"Expected keys 'X' and 'y' in {path}. Found keys: {list(mat.keys())}")
    y = y.ravel().astype(int)
    return X.astype(float), y

X_all, y_all = load_shuttle_mat("shuttle.mat")

def fit_all_models(X_train, y_train, seed=0, dif_hidden=(64, 32), loda_bins=30, use_dif=True):
    contamination = float(np.mean(y_train))

    iforest = IForest(contamination=contamination, random_state=seed).fit(X_train)

    # aggressively optimized - reduced parameters for shuttle dataset
    if use_dif:
        try:
            dif = DIF(contamination=contamination,
                      hidden_neurons=dif_hidden,
                      random_state=seed,
                      n_ensemble=2,
                      n_estimators=10,
                      epochs=5
                      ).fit(X_train)
        except TypeError:
            try:
                dif = DIF(contamination=contamination, 
                          random_state=seed,
                          n_ensemble=2,
                          n_estimators=10).fit(X_train)
            except:
                print("DIF failed, skipping...")
                dif = None
    else:
        dif = None

    loda = LODA(contamination=contamination, n_bins=loda_bins).fit(X_train)

    models = {"IForest": iforest, "LODA": loda}
    if dif is not None:
        models["DIF"] = dif
    
    return models

def eval_models(models, X_test, y_test):
    out = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        scores = model.decision_function(X_test)
        ba = balanced_accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, scores)
        except ValueError:
            auc = np.nan
        out[name] = (ba, auc)
    return out

SEEDS = range(10)
results = { "IForest": {"BA": [], "AUC": []},
            "DIF":     {"BA": [], "AUC": []},
            "LODA":    {"BA": [], "AUC": []} }

print("Starting evaluation with 10 different train-test splits...")
for i, seed in enumerate(SEEDS):
    print(f"\nSplit {i+1}/10 (seed={seed})...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.40, random_state=seed, stratify=y_all
    )

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = fit_all_models(X_tr_s, y_tr, seed=seed, dif_hidden=(32, 16), loda_bins=30, use_dif=True)
    scores = eval_models(models, X_te_s, y_te)

    for name, (ba, auc) in scores.items():
        results[name]["BA"].append(ba)
        results[name]["AUC"].append(auc)
    
    print(f"  Split {i+1} completed.")

def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    return np.nanmean(arr), np.nanstd(arr)

print("\n=== Shuttle (ODDS) — 10 splits, 40% test ===")
for name in ["IForest", "DIF", "LODA"]:
    if len(results[name]["BA"]) > 0:
        ba_mean, ba_std = mean_std(results[name]["BA"])
        auc_mean, auc_std = mean_std(results[name]["AUC"])
        print(f"{name:8s}  BA:  {ba_mean:.3f} ± {ba_std:.3f}   ROC-AUC: {auc_mean:.3f} ± {auc_std:.3f}")
    else:
        print(f"{name:8s}  No results available (model skipped)")