"""
GFP variants dataset: step 4 — broad model comparison with hyperparameter search.

Eight model families are compared on identical features and identical
test sets. Because they do not all scale the same way, the comparison
runs in two tiers:

  tier "full"   trains on the whole training split (~113k rows, random
                split). Only the models that scale to that size.
  tier "sub12k" trains on a fixed 12,000-row subsample of the SAME
                training split, and includes every model -- notably
                kernel SVR (whose kernel matrix is quadratic in sample
                count) and RandomForest (slow on 6,625 sparse features).

Both tiers are scored on the FULL val/test sets of their split, so tier
results are directly comparable and the cost of subsampling is visible.

Hyperparameters are selected on the validation split by Spearman ρ.
Test metrics are recorded for every config but never used for selection.

This script is resume-safe: results are appended to model_comparison.csv
after every single config, and configs already present are skipped. It
stops launching new configs after TIME_BUDGET seconds so it can be
re-run repeatedly until complete.

Run with: python compare_models.py --split random   [--tier full|sub12k]
          python compare_models.py --split position
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"
RESULTS_CSV = OUT_DIR / "model_comparison.csv"

SPLIT_COLUMN = {"random": "split_random", "position": "split_position_holdout"}
SUBSAMPLE_N = 12_000
SVR_SUBSAMPLE_N = 6_000   # kernel SVR only; see note in the fit loop
SUBSAMPLE_SEED = 42
# A single fit cannot be checkpointed, so any config that takes longer than
# the ~120s tool limit can never complete. Budget is set low enough that a
# long config started at the deadline still finishes inside that limit.
TIME_BUDGET = 30          # stop starting new configs after this many seconds


def model_grid(tier: str) -> list[tuple[str, str, object]]:
    """(model_name, config_id, estimator) triples to evaluate."""
    grid: list[tuple[str, str, object]] = []

    for a in [0.1, 1.0, 10.0, 100.0]:
        grid.append(("ridge", f"alpha={a}", Ridge(alpha=a, solver="lsqr")))

    # exact coordinate-descent Lasso needs ~100s per fit on the full
    # matrix, so L1/elastic-net use SGD, which is the standard solver at
    # this scale and gives the same penalty
    for a in [1e-6, 1e-5, 1e-4, 1e-3]:
        grid.append(("lasso", f"alpha={a}", SGDRegressor(
            penalty="l1", alpha=a, max_iter=40, tol=1e-4, random_state=0)))

    for a in [1e-5, 1e-4]:
        for r in [0.15, 0.5, 0.85]:
            grid.append(("elastic_net", f"alpha={a},l1={r}", SGDRegressor(
                penalty="elasticnet", alpha=a, l1_ratio=r,
                max_iter=40, tol=1e-4, random_state=0)))

    # liblinear needs more iterations as C grows; capped so no single fit
    # can exceed the wall-clock limit
    for c in [0.1, 1.0, 10.0]:
        grid.append(("linear_svr", f"C={c}", LinearSVR(
            C=c, max_iter=1000, random_state=0)))

    for d in [6, 8]:
        for lr in [0.05, 0.1]:
            for n in [400]:
                grid.append((
                    "xgboost", f"depth={d},lr={lr},n={n}",
                    xgb.XGBRegressor(
                        n_estimators=n, max_depth=d, learning_rate=lr,
                        subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", n_jobs=-1, random_state=0)))

    # NOTE: HistGradientBoosting is deliberately absent. It cannot accept
    # sparse input, and densifying 113k x 6,625 needs ~3 GB. XGBoost is the
    # gradient-boosting representative on these features; the HGB result on
    # the compact dense block is reported separately in step 3.

    # MLP, RandomForest and kernel SVR are subsample-only. On 113k rows a
    # single MLP fit needs ~7 minutes and RF/SVR far longer, which cannot
    # complete here -- so they are compared fairly against everything else
    # inside the sub12k tier instead.
    if tier == "sub12k":
        # max_iter capped at 15: at 25 the (512,256) net exceeds the
        # wall-clock limit on the position split. early_stopping is on, so
        # this is an upper bound the smaller nets rarely reach.
        for hidden in [(256,), (512, 256)]:
            for a in [1e-4, 1e-3]:
                grid.append((
                    "mlp", f"hidden={hidden},alpha={a}",
                    MLPRegressor(hidden_layer_sizes=hidden, alpha=a,
                                 max_iter=15, early_stopping=True,
                                 random_state=0)))

        for mf in ["sqrt", 0.05]:
            for depth in [None, 20]:
                grid.append((
                    "random_forest", f"maxfeat={mf},depth={depth}",
                    RandomForestRegressor(
                        n_estimators=100, max_features=mf, max_depth=depth,
                        n_jobs=-1, random_state=0)))

        # gamma is fixed to 'scale': a larger gamma (e.g. 0.01) makes the
        # kernel much more local on this data, which explodes the support-
        # vector count and the fit time past the wall-clock limit
        # C is also capped: at C=10 the optimiser no longer converges inside
        # the wall-clock limit, so the SVR search covers C <= 1 only.
        for c in [0.1, 1.0]:
            grid.append((
                "svr_rbf", f"C={c},gamma=scale",
                SVR(C=c, gamma="scale", cache_size=1000)))

    return grid


def metrics(y_true, y_pred, prefix: str) -> dict:
    return {
        f"{prefix}_spearman": spearmanr(y_true, y_pred).statistic,
        f"{prefix}_pearson": pearsonr(y_true, y_pred)[0],
        f"{prefix}_rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        f"{prefix}_r2": 1 - np.sum((y_true - y_pred) ** 2) / np.sum(
            (y_true - y_true.mean()) ** 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "position"], required=True)
    parser.add_argument("--tier", choices=["full", "sub12k"], default=None,
                        help="default: run both tiers")
    args = parser.parse_args()

    X_sparse = sparse.load_npz(FEAT_DIR / "X_sparse.npz")
    X_dense = np.load(FEAT_DIR / "X_dense.npy")
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")

    split = meta[SPLIT_COLUMN[args.split]].to_numpy()
    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    scaler = StandardScaler().fit(X_dense[tr])
    X_dense_s = scaler.transform(X_dense).astype(np.float32)
    X = sparse.hstack([X_sparse, sparse.csr_matrix(X_dense_s)]).tocsr()

    X_tr_full, y_tr_full = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    sub_idx = rng.choice(X_tr_full.shape[0],
                         min(SUBSAMPLE_N, X_tr_full.shape[0]), replace=False)

    done = set()
    if RESULTS_CSV.exists():
        prev = pd.read_csv(RESULTS_CSV)
        done = set(zip(prev["split"], prev["tier"], prev["model"], prev["config"]))

    tiers = [args.tier] if args.tier else ["full", "sub12k"]
    started = time.time()

    for tier in tiers:
        X_tr = X_tr_full if tier == "full" else X_tr_full[sub_idx]
        y_tr = y_tr_full if tier == "full" else y_tr_full[sub_idx]

        for model_name, config, estimator in model_grid(tier):
            key = (args.split, tier, model_name, config)
            if key in done:
                continue
            if time.time() - started > TIME_BUDGET:
                print(f"\ntime budget reached — re-run to continue "
                      f"(next up: {tier}/{model_name}/{config})")
                return

            # Kernel SVR is quadratic to fit AND its prediction cost scales
            # with (support vectors x rows to predict), which blows past the
            # limit on the position split's 57k val+test rows. It therefore
            # gets a smaller training subsample than the rest; n_train in the
            # results records this so the handicap stays visible.
            if model_name == "svr_rbf" and tier == "sub12k":
                X_fit, y_fit = X_tr[:SVR_SUBSAMPLE_N], y_tr[:SVR_SUBSAMPLE_N]
            else:
                X_fit, y_fit = X_tr, y_tr

            t0 = time.time()
            estimator.fit(X_fit, y_fit)
            row = {
                "split": args.split, "tier": tier, "model": model_name,
                "config": config, "n_train": X_fit.shape[0],
                "fit_seconds": round(time.time() - t0, 1),
            }
            row.update(metrics(y_va, estimator.predict(X_va), "val"))
            row.update(metrics(y_te, estimator.predict(X_te), "test"))

            # append immediately, so a timeout never loses completed work
            pd.DataFrame([row]).to_csv(
                RESULTS_CSV, mode="a", header=not RESULTS_CSV.exists(), index=False)
            print(f"{tier:7s} {model_name:14s} {config:22s} "
                  f"val_rho={row['val_spearman']:.4f} "
                  f"test_rho={row['test_spearman']:.4f} ({row['fit_seconds']}s)",
                  flush=True)

    print("\nall configs complete for split=" + args.split)


if __name__ == "__main__":
    main()
