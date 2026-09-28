"""
GFP variants dataset: step 10a — is brightness predictable *among functional
variants*, or only dead-vs-alive?

Why this exists. The design step (step 10) needs to rank candidates by
brightness. Checking the step-9 out-of-fold predictions before trusting them
showed that the ~0.45 Spearman is almost entirely dead/alive separation:

    overall                       rho  0.455      (AUC alive = 0.778)
    functional only (y >= 0.5)    rho  0.247
    y >= 0.8                      rho  0.130
    y >= 1.0                      rho -0.146
    variants with y >= 1.2        median predicted rank 6452 / 16017

and the top-20 of the ranked list contains nothing above 1.2x WT, i.e.
enrichment below 1. A design loop that maximises this model's output is
optimising a quantity that is uncorrelated with truth in exactly the regime
that matters.

This is the project's open item #3 ("two-stage dark/bright classifier +
regressor"), promoted to blocking, because whether the design step can rank
brightness at all depends on the answer.

Two questions, on the settled 5-fold position CV (SEED=7, folds imported from
evo_benchmark so they are identical to steps 5, 8 and 9):

  1. Does a dedicated classifier beat the regressor at dead/alive?
  2. Does a regressor trained ONLY on functional rows rank brightness among
     functional rows better than the pooled regressor does? If the pooled
     model is merely distracted by the dead spike, this should help. If
     brightness among functional variants is simply not a function of these
     features, it will not.

Stage 2 is scored on functional TEST rows only, against the step-9 model's
predictions restricted to the same rows, so the comparison is paired.

Resume-safe: one npz per fold.

Run with: python two_stage_benchmark.py
Output: output/extrap_cv/twostage_fold{k}.npz, output/two_stage_results.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# identical folds to steps 5/8/9 by construction
import importlib, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
_evo_benchmark = importlib.import_module("18_evo_benchmark")
assign_position_folds = _evo_benchmark.assign_position_folds
fold_masks = _evo_benchmark.fold_masks
N_FOLDS = _evo_benchmark.N_FOLDS
SEED = _evo_benchmark.SEED
XGB_CFG = _evo_benchmark.XGB_CFG

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"
CV_DIR = OUT_DIR / "extrap_cv"

DEAD_THRESHOLD = 0.5
TIME_BUDGET = 480


def run_folds():
    X_dense = np.load(FEAT_DIR / "X_dense.npy")
    X_evo = np.load(FEAT_DIR / "X_evo.npy")
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")

    rng = np.random.default_rng(SEED)
    folds, positions = assign_position_folds(meta, rng)

    started = time.time()
    for k in range(N_FOLDS):
        path = CV_DIR / f"twostage_fold{k}.npz"
        if path.exists():
            continue
        if time.time() - started > TIME_BUDGET:
            print(f"time budget reached — re-run to continue (next: fold {k})")
            return False
        train, test, n_mut = fold_masks(meta, positions, folds, k)

        combined = np.hstack([X_dense, X_evo])
        scaler = StandardScaler().fit(combined[train])
        X = scaler.transform(combined).astype(np.float32)

        alive = y >= DEAD_THRESHOLD
        t0 = time.time()

        clf = xgb.XGBClassifier(tree_method="hist", n_jobs=-1, random_state=0,
                                eval_metric="logloss", **XGB_CFG)
        clf.fit(X[train], alive[train].astype(int))
        p_alive = clf.predict_proba(X[test])[:, 1]

        fit_rows = train & alive
        reg = xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0,
                               **XGB_CFG)
        reg.fit(X[fit_rows], y[fit_rows])
        pred_bright = reg.predict(X[test])

        np.savez_compressed(
            path, p_alive=p_alive, pred_bright=pred_bright, y_true=y[test],
            n_mut=n_mut[test], backbone=meta.loc[test, "Backbone"].to_numpy())
        print(f"fold {k}: train={train.sum():6d} (functional {fit_rows.sum():6d}) "
              f"test={test.sum():5d}  ({time.time() - t0:.0f}s)", flush=True)
    return True


def pooled(prefix, keys):
    out = {k: [] for k in keys}
    for k in range(N_FOLDS):
        d = np.load(CV_DIR / prefix.format(k=k), allow_pickle=True)
        for key in keys:
            out[key].append(d[key])
    return {k: np.concatenate(v) for k, v in out.items()}


def analyse():
    ts = pooled("twostage_fold{k}.npz",
                ["p_alive", "pred_bright", "y_true", "n_mut", "backbone"])
    base = pooled("evo_fold{k}.npz", ["pred", "y_true", "n_mut"])

    # the two files must describe the same rows in the same order
    assert np.allclose(ts["y_true"], base["y_true"]), "fold rows disagree"

    y = ts["y_true"]
    alive = y >= DEAD_THRESHOLD
    rows = []

    rows.append({
        "question": "1. dead/alive", "model": "step-9 regressor (as classifier)",
        "subset": "all novel", "n": len(y),
        "metric": "AUC", "value": roc_auc_score(alive, base["pred"])})
    rows.append({
        "question": "1. dead/alive", "model": "dedicated classifier",
        "subset": "all novel", "n": len(y),
        "metric": "AUC", "value": roc_auc_score(alive, ts["p_alive"])})

    for label, mask in [("functional (y>=0.5)", alive),
                        ("y>=0.8", y >= 0.8),
                        ("y>=1.0", y >= 1.0)]:
        rows.append({
            "question": "2. brightness among functional",
            "model": "step-9 regressor (pooled fit)", "subset": label,
            "n": int(mask.sum()), "metric": "Spearman",
            "value": spearmanr(base["pred"][mask], y[mask]).statistic})
        rows.append({
            "question": "2. brightness among functional",
            "model": "regressor fit on functional only", "subset": label,
            "n": int(mask.sum()), "metric": "Spearman",
            "value": spearmanr(ts["pred_bright"][mask], y[mask]).statistic})

    # does the combined two-stage score find bright variants any better?
    combined = ts["p_alive"] * ts["pred_bright"]
    for name, score in [("step-9 regressor", base["pred"]),
                        ("functional-only regressor", ts["pred_bright"]),
                        ("p_alive x brightness", combined)]:
        order = np.argsort(-score)
        for k in [20, 100]:
            sel = y[order[:k]]
            rows.append({
                "question": f"3. precision@{k} for y>=1.2", "model": name,
                "subset": "all novel", "n": len(y),
                "metric": f"frac>=1.2 (baseline {(y >= 1.2).mean():.3f})",
                "value": float((sel >= 1.2).mean())})

    df = pd.DataFrame(rows)
    df["value"] = df["value"].round(4)
    return df


def main():
    CV_DIR.mkdir(parents=True, exist_ok=True)
    if not run_folds():
        return
    df = analyse()
    df.to_csv(OUT_DIR / "two_stage_results.csv", index=False)
    for q, block in df.groupby("question"):
        print(f"\n{q}")
        print(block[["model", "subset", "n", "metric", "value"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
