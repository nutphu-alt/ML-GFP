"""
GFP variants dataset: step 3b — train and evaluate baseline models.

Models (all predicting fold-WT brightness):

  mean_by_backbone  per-backbone training mean. The floor: any model that
                    cannot beat this has learned nothing.
  ridge_sparse      Ridge on the (backbone, position, mutant AA) indicators.
                    The classic additive fitness-landscape model.
  ridge_dense       Ridge on the compact transferable features only.
  ridge_combined    Ridge on both blocks.
  hgb_dense         HistGradientBoosting on the compact features; picks up
                    non-linearity / epistasis the additive models can't.

Each is evaluated on whichever split is requested. Alpha for the ridge
models is chosen on the validation split, never on test.

Run with: python train_baselines.py --split random
          python train_baselines.py --split position

Input : output/features/*
Output: output/metrics_<split>.csv, output/preds_<split>.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"

SPLIT_COLUMN = {"random": "split_random", "position": "split_position_holdout"}
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]


def load_features():
    X_sparse = sparse.load_npz(FEAT_DIR / "X_sparse.npz")
    X_dense = np.load(FEAT_DIR / "X_dense.npy")
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    return X_sparse, X_dense, y, meta


def evaluate(y_true, y_pred, backbones) -> dict:
    """Overall + per-backbone metrics. Spearman is the headline number:
    it is what matters for ranking variants, and it is invariant to the
    per-library scale differences this dataset is known to have."""
    out = {
        "spearman": spearmanr(y_true, y_pred).statistic,
        "pearson": pearsonr(y_true, y_pred)[0],
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2),
    }
    for backbone in sorted(pd.unique(backbones)):
        mask = backbones == backbone
        if mask.sum() > 10:
            out[f"spearman_{backbone}"] = spearmanr(y_true[mask], y_pred[mask]).statistic
    return out


def fit_ridge(X_tr, y_tr, X_va, y_va, X_te, sparse_input: bool):
    """Fit Ridge, choosing alpha on the validation split."""
    # lsqr converges far faster than sparse_cg on the combined block, which
    # mixes a very sparse indicator matrix with a dense standardised one
    best = (None, -np.inf, None)
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, solver="lsqr" if sparse_input else "auto")
        model.fit(X_tr, y_tr)
        score = spearmanr(y_va, model.predict(X_va)).statistic
        if score > best[1]:
            best = (alpha, score, model)
    alpha, _, model = best
    return model.predict(X_te), model.predict(X_va), alpha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "position"], required=True)
    args = parser.parse_args()

    split_col = SPLIT_COLUMN[args.split]
    X_sparse, X_dense, y, meta = load_features()

    split = meta[split_col].to_numpy()
    backbone = meta["Backbone"].to_numpy()
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    print(f"split={args.split}  train={tr.sum()}  val={va.sum()}  test={te.sum()}")

    # backbone one-hot lives in columns 1..4 of the dense block
    backbone_oh = X_dense[:, 1:5]
    X_sparse_plus = sparse.hstack([X_sparse, sparse.csr_matrix(backbone_oh)]).tocsr()

    scaler = StandardScaler().fit(X_dense[tr])
    X_dense_s = scaler.transform(X_dense)
    X_combined = sparse.hstack([X_sparse_plus, sparse.csr_matrix(X_dense_s)]).tocsr()

    results, preds = {}, {}

    # ---- floor: per-backbone training mean -------------------------------
    pred = np.zeros(te.sum(), dtype=np.float32)
    te_backbone = backbone[te]
    for b in np.unique(backbone):
        train_mean = y[tr & (backbone == b)].mean()
        pred[te_backbone == b] = train_mean
    results["mean_by_backbone"] = evaluate(y[te], pred, te_backbone)
    preds["mean_by_backbone"] = pred

    # ---- ridge variants ---------------------------------------------------
    for name, X, is_sparse in [
        ("ridge_sparse", X_sparse_plus, True),
        ("ridge_dense", X_dense_s, False),
        ("ridge_combined", X_combined, True),
    ]:
        t0 = time.time()
        pred_te, _, alpha = fit_ridge(X[tr], y[tr], X[va], y[va], X[te], sparse_input=is_sparse)
        results[name] = evaluate(y[te], pred_te, te_backbone)
        results[name]["alpha"] = alpha
        results[name]["fit_seconds"] = round(time.time() - t0, 1)
        preds[name] = pred_te
        print(f"{name:16s} alpha={alpha:<6} spearman={results[name]['spearman']:.4f} "
              f"({time.time() - t0:.0f}s)")

    # ---- gradient boosting on the compact features -----------------------
    t0 = time.time()
    hgb = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.1, early_stopping=True,
        validation_fraction=0.1, random_state=42,
    )
    hgb.fit(X_dense[tr], y[tr])
    pred_te = hgb.predict(X_dense[te])
    results["hgb_dense"] = evaluate(y[te], pred_te, te_backbone)
    results["hgb_dense"]["fit_seconds"] = round(time.time() - t0, 1)
    results["hgb_dense"]["n_iter"] = int(hgb.n_iter_)
    preds["hgb_dense"] = pred_te
    print(f"{'hgb_dense':16s} iters={hgb.n_iter_:<6} "
          f"spearman={results['hgb_dense']['spearman']:.4f} ({time.time() - t0:.0f}s)")

    table = pd.DataFrame(results).T
    table.index.name = "model"
    table.to_csv(OUT_DIR / f"metrics_{args.split}.csv")
    np.savez_compressed(OUT_DIR / f"preds_{args.split}.npz", y_true=y[te], **preds)

    print(f"\n=== {args.split} split ===")
    cols = [c for c in ["spearman", "pearson", "rmse", "r2"] if c in table.columns]
    print(table[cols].round(4).to_string())
    print(f"\nwrote metrics_{args.split}.csv")


if __name__ == "__main__":
    main()
