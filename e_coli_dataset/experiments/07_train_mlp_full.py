"""
GFP variants dataset: step 4c — MLP on the FULL training set.

The step-4 comparison could only fit the MLP on a 12,000-row subsample,
because a single MLPRegressor.fit() on all ~113k rows needs ~9-15 minutes
(measured: ~36s per epoch at 6,625 inputs with hidden=(512,256)) and a
fit cannot be interrupted and resumed.

This script works around that by driving the epochs itself:

  * one partial_fit pass per epoch, reshuffled each time
  * after every epoch, score the REAL validation split (better than
    MLPRegressor's internal early stopping, which carves its holdout out
    of the training data at random)
  * keep a copy of the weights from the best epoch so far
  * stop on a wall-clock budget, on early-stopping patience, or at
    MAX_EPOCHS -- and checkpoint to disk either way

Re-running resumes from the checkpoint, so the full fit can span as many
invocations as it needs. Final metrics are appended to
model_comparison.csv with tier="full" so the leaderboard picks them up.

Run with: python train_mlp_full.py --split random  --alpha 0.0001
          python train_mlp_full.py --split position --alpha 0.001
"""

import argparse
import copy
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"
CKPT_DIR = OUT_DIR / "mlp_checkpoints"
RESULTS_CSV = OUT_DIR / "model_comparison.csv"

SPLIT_COLUMN = {"random": "split_random", "position": "split_position_holdout"}
HIDDEN = (512, 256)          # best architecture from the sub12k search
MAX_EPOCHS = 15              # matches the max_iter used in the sub12k grid
PATIENCE = 4                 # epochs without val improvement before stopping
TIME_BUDGET = 90             # seconds. The host kills calls at ~180s, and an
                             # epoch costs ~45s, so this stops after ~2 epochs
                             # with room to checkpoint and exit cleanly.


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
    # Default alpha follows the split, per the module docstring above: the
    # sub12k search picked 1e-4 for random and 1e-3 for position. A single
    # default meant `--split position` silently trained a different model from
    # the documented one, under a checkpoint name 09 does not look for.
    parser.add_argument("--alpha", type=float, default=None,
                        help="L2 penalty; defaults to 1e-4 (random) / 1e-3 (position)")
    args = parser.parse_args()
    if args.alpha is None:
        args.alpha = {"random": 1e-4, "position": 1e-3}[args.split]
        print(f"alpha not given — using the documented {args.alpha:g} for "
              f"--split {args.split}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"mlp_{args.split}_alpha{args.alpha}.joblib"

    X_sparse = sparse.load_npz(FEAT_DIR / "X_sparse.npz")
    X_dense = np.load(FEAT_DIR / "X_dense.npy")
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")

    split = meta[SPLIT_COLUMN[args.split]].to_numpy()
    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    scaler = StandardScaler().fit(X_dense[tr])
    X_dense_s = scaler.transform(X_dense).astype(np.float32)
    X = sparse.hstack([X_sparse, sparse.csr_matrix(X_dense_s)]).tocsr()

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]
    print(f"split={args.split} alpha={args.alpha} "
          f"train={X_tr.shape[0]} val={X_va.shape[0]} test={X_te.shape[0]}",
          flush=True)

    if ckpt_path.exists():
        state = joblib.load(ckpt_path)
        model, epoch = state["model"], state["epoch"]
        best_val, best_weights = state["best_val"], state["best_weights"]
        stale = state["stale"]
        print(f"resumed from checkpoint at epoch {epoch} "
              f"(best val rho={best_val:.4f})", flush=True)
    else:
        model = MLPRegressor(
            hidden_layer_sizes=HIDDEN, alpha=args.alpha,
            learning_rate_init=1e-3, random_state=0)
        epoch, best_val, best_weights, stale = 0, -np.inf, None, 0

    rng = np.random.default_rng(1000 + epoch)
    started = time.time()

    while epoch < MAX_EPOCHS and stale < PATIENCE:
        if time.time() - started > TIME_BUDGET:
            print("time budget reached — re-run to continue", flush=True)
            break

        order = rng.permutation(X_tr.shape[0])       # reshuffle every epoch
        t0 = time.time()
        model.partial_fit(X_tr[order], y_tr[order])
        epoch += 1

        val_rho = spearmanr(y_va, model.predict(X_va)).statistic
        if val_rho > best_val:
            best_val, stale = val_rho, 0
            best_weights = (copy.deepcopy(model.coefs_),
                            copy.deepcopy(model.intercepts_))
            marker = " *best"
        else:
            stale += 1
            marker = ""
        print(f"  epoch {epoch:2d}  val_rho={val_rho:.4f}  "
              f"({time.time() - t0:.0f}s){marker}", flush=True)

        joblib.dump({"model": model, "epoch": epoch, "best_val": best_val,
                     "best_weights": best_weights, "stale": stale}, ckpt_path)

    finished = epoch >= MAX_EPOCHS or stale >= PATIENCE
    if not finished:
        return

    # restore the best epoch's weights before scoring the test set
    model.coefs_, model.intercepts_ = best_weights
    row = {
        "split": args.split, "tier": "full", "model": "mlp",
        "config": f"hidden={HIDDEN},alpha={args.alpha}",
        "n_train": int(X_tr.shape[0]), "fit_seconds": None, "epochs": epoch,
    }
    row.update(metrics(y_va, model.predict(X_va), "val"))
    row.update(metrics(y_te, model.predict(X_te), "test"))

    prev = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()
    key = (prev["split"].eq(args.split) & prev["tier"].eq("full")
           & prev["model"].eq("mlp") & prev["config"].eq(row["config"])) \
        if len(prev) else pd.Series(dtype=bool)
    if len(prev) and key.any():
        prev = prev[~key]
    pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(
        RESULTS_CSV, index=False)

    print(f"\nDONE after {epoch} epochs — "
          f"val_rho={row['val_spearman']:.4f} test_rho={row['test_spearman']:.4f} "
          f"test_r2={row['test_r2']:.4f}")


if __name__ == "__main__":
    main()
