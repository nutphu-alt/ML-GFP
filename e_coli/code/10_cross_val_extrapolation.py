"""
GFP variants dataset: step 5 — a proper extrapolation benchmark.

Step 4d found that XGBoost may be the best model at never-assayed
positions, but the evidence was 293 single-mutant rows and the bootstrap
CI crossed zero. That benchmark was small only because the position
holdout withheld 6% of positions, so just 6% of single mutants landed in
test — while the dataset actually contains 4,596 of them.

This script uses all of them. Positions are split into K folds per
backbone; for each fold:

    train = rows whose mutations ALL sit at positions outside the fold
    test  = rows whose mutations ALL sit at positions INSIDE the fold

Rows straddling the boundary are dropped from both, so no training row
ever touches a held-out position. Every variant is tested exactly once,
by a model that never saw any of its positions — giving out-of-fold
predictions for the full single-mutant set rather than a 6% slice.

Resume-safe: per (fold, model) predictions are written to disk as they
finish, and completed work is skipped on re-run.

Run with: python cross_val_extrapolation.py [--models ridge,xgboost,mlp]
Output: output/extrap_cv/fold{K}_{model}.npz, then analysed by
        analyze_extrapolation_cv.py
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"
CV_DIR = OUT_DIR / "extrap_cv"

MUT_RE = re.compile(r"[A-Z](\d+)[A-Z]")
N_FOLDS = 5
SEED = 7
TIME_BUDGET = 100          # stop starting new work after this many seconds

# MLP settings mirror step 4c but with a shorter ceiling: the full-data runs
# early-stopped at epoch 6, and each fold trains on roughly half the data.
MLP_MAX_EPOCHS = 8
MLP_PATIENCE = 3


def assign_position_folds(meta: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Split each backbone's mutated positions into N_FOLDS disjoint sets."""
    positions = meta["Variant name"].apply(
        lambda n: [int(p) for p in MUT_RE.findall(str(n))])
    folds = {}
    for backbone in sorted(meta["Backbone"].unique()):
        mask = meta["Backbone"] == backbone
        all_pos = sorted({p for ps in positions[mask] for p in ps})
        arr = np.array(all_pos)
        rng.shuffle(arr)
        folds[backbone] = np.array_split(arr, N_FOLDS)
    return folds, positions


def fold_masks(meta, positions, folds, k):
    """train = touches no fold-k position; test = touches ONLY fold-k positions."""
    held = {b: set(f[k].tolist()) for b, f in folds.items()}
    n_novel, n_mut = [], []
    for b, ps in zip(meta["Backbone"], positions):
        h = held[b]
        n_novel.append(sum(p in h for p in ps))
        n_mut.append(len(ps))
    n_novel, n_mut = np.array(n_novel), np.array(n_mut)

    train = (n_novel == 0) & (n_mut > 0)
    test = (n_novel == n_mut) & (n_mut > 0)
    return train, test, n_mut


def fit_mlp(X_tr, y_tr, X_te, fold, started):
    """Epoch loop with a per-fold checkpoint.

    Architecture matches step 4c — (512,256), alpha=1e-3 — so this is the
    same model that won the headline comparison, not a cheaper stand-in.
    One epoch on ~70k rows costs ~30s, so a whole fold cannot finish inside
    one run; state is saved after every epoch and the caller re-runs.
    Returns None while still training, predictions once converged.
    """
    import joblib
    from scipy.stats import spearmanr

    ck_path = CV_DIR / f"fold{fold}_mlp.ckpt"
    if ck_path.exists():
        st = joblib.load(ck_path)
        model, epoch, best, stale = st["model"], st["epoch"], st["best"], st["stale"]
        best_state, fit_idx, val_idx = st["best_state"], st["fit_idx"], st["val_idx"]
    else:
        model = MLPRegressor(hidden_layer_sizes=(512, 256), alpha=1e-3,
                             learning_rate_init=1e-3, random_state=0)
        # a fixed 10% of this fold's training rows picks the stopping epoch
        idx = np.random.default_rng(100 + fold).permutation(X_tr.shape[0])
        cut = int(0.9 * len(idx))
        fit_idx, val_idx = idx[:cut], idx[cut:]
        epoch, best, stale, best_state = 0, -np.inf, 0, None

    while epoch < MLP_MAX_EPOCHS and stale < MLP_PATIENCE:
        if time.time() - started > TIME_BUDGET:
            joblib.dump({"model": model, "epoch": epoch, "best": best,
                         "stale": stale, "best_state": best_state,
                         "fit_idx": fit_idx, "val_idx": val_idx}, ck_path)
            print(f"  fold {fold} mlp paused at epoch {epoch} "
                  f"(best {best:.4f}) — re-run to continue", flush=True)
            return None

        model.partial_fit(X_tr[fit_idx], y_tr[fit_idx])
        epoch += 1
        score = spearmanr(y_tr[val_idx], model.predict(X_tr[val_idx])).statistic
        if score > best:
            best, stale = score, 0
            best_state = ([c.copy() for c in model.coefs_],
                          [b.copy() for b in model.intercepts_])
        else:
            stale += 1
        joblib.dump({"model": model, "epoch": epoch, "best": best,
                     "stale": stale, "best_state": best_state,
                     "fit_idx": fit_idx, "val_idx": val_idx}, ck_path)

    model.coefs_, model.intercepts_ = best_state
    return model.predict(X_te)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="ridge,xgboost,mlp")
    args = parser.parse_args()
    wanted = args.models.split(",")

    CV_DIR.mkdir(parents=True, exist_ok=True)

    X_sparse = sparse.load_npz(FEAT_DIR / "X_sparse.npz")
    X_dense = np.load(FEAT_DIR / "X_dense.npy")
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")

    rng = np.random.default_rng(SEED)
    folds, positions = assign_position_folds(meta, rng)

    started = time.time()
    for k in range(N_FOLDS):
        train, test, n_mut = fold_masks(meta, positions, folds, k)

        # scaler is fit on this fold's training rows only
        scaler = StandardScaler().fit(X_dense[train])
        X = sparse.hstack([X_sparse,
                           sparse.csr_matrix(scaler.transform(X_dense).astype(np.float32))
                           ]).tocsr()
        X_tr, y_tr, X_te = X[train], y[train], X[test]

        for name in wanted:
            path = CV_DIR / f"fold{k}_{name}.npz"
            if path.exists():
                continue
            if time.time() - started > TIME_BUDGET:
                print(f"time budget reached — re-run to continue "
                      f"(next: fold {k}, {name})")
                return

            t0 = time.time()
            if name == "ridge":
                pred = Ridge(alpha=10.0, solver="lsqr").fit(X_tr, y_tr).predict(X_te)
            elif name == "xgboost":
                pred = xgb.XGBRegressor(
                    n_estimators=400, max_depth=8, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                    n_jobs=-1, random_state=0).fit(X_tr, y_tr).predict(X_te)
            elif name == "mlp":
                pred = fit_mlp(X_tr, y_tr, X_te, k, started)
                if pred is None:      # paused mid-fold; resume on next run
                    return
            else:
                raise ValueError(name)

            np.savez_compressed(
                path, pred=pred, y_true=y[test], n_mut=n_mut[test],
                backbone=meta.loc[test, "Backbone"].to_numpy(),
                row_index=np.flatnonzero(test))
            print(f"fold {k} {name:8s} train={train.sum():6d} test={test.sum():5d} "
                  f"single={int((n_mut[test] == 1).sum()):4d} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print("\nall folds complete")


if __name__ == "__main__":
    main()
