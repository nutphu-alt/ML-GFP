"""
GFP variants dataset: step 8 -- retune XGBoost specifically for extrapolation.

Every XGBoost result so far (steps 4, 5, 7) used one fixed configuration
(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.8,
colsample_bytree=0.8) that was never actually chosen for this job -- it was
carried over from the step-4 model-comparison grid, which selected configs
by validation Spearman on the RANDOM split. That is the split where XGBoost
performs worst. A config picked for interpolation has never been tuned for
the regime where XGBoost is actually the best model: unseen positions.

Two things follow from step 7 as well:
  - the 6,559 sparse position/backbone indicators contribute ~nothing once
    positions are unseen (dense 0.481 vs combined 0.483 mean Spearman), so
    this search drops them entirely and tunes on the 66 dense descriptors
    only -- faster fits, no measurable cost in this regime.
  - the right objective to tune against is out-of-fold single-mutant
    Spearman (the step-5 benchmark, n=4,596), not the random-split
    validation score.

Two phases, resume-safe, run with the SAME fold assignment as step 5
(assign_position_folds, SEED=7) so results are directly comparable:

  --phase grid   sweep candidate configs on folds 0-1 only (cheap: dense
                 features, 2 of 5 folds). Ranks configs by mean single-
                 mutant Spearman. Writes output/xgb_tune_grid.csv.
  --phase eval   take the current default plus the top candidates from the
                 grid, run them across all 5 folds, pool out-of-fold
                 predictions, and paired-bootstrap each candidate against
                 the default. Writes output/xgb_tune_eval_fold{k}.npz and
                 prints/saves output/xgb_tune_results.csv.

Run with: python tune_xgb_extrapolation.py --phase grid
          python tune_xgb_extrapolation.py --phase eval
"""

import argparse
import itertools
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.name == "code") else Path(".")
# fall back to explicit project layout when run from the scratch outputs dir
CANDIDATE_BASE = Path(r"/sessions/bold-admiring-volta/mnt/ML for GFP/e_coli")
if CANDIDATE_BASE.exists():
    BASE_DIR = CANDIDATE_BASE
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"
CV_DIR = OUT_DIR / "extrap_cv"

MUT_RE = re.compile(r"[A-Z](\d+)[A-Z]")
N_FOLDS = 5
SEED = 7                 # must match cross_val_extrapolation.py exactly
GRID_FOLDS = [0, 1]       # cheap subset for the sweep phase
TIME_BUDGET = 150

DEFAULT_CFG = dict(n_estimators=400, max_depth=8, learning_rate=0.1,
                   subsample=0.8, colsample_bytree=0.8)

# a deliberately narrow grid around parameters that matter most for
# generalisation to unseen positions: shallower trees and more
# regularisation are expected to transfer better than the interpolation-
# tuned default (deep trees can memorise the training positions).
GRID = {
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [300, 600],
    "min_child_weight": [1, 5],
}


def assign_position_folds(meta: pd.DataFrame, rng: np.random.Generator):
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


def load_data():
    X_dense = np.load(FEAT_DIR / "X_dense.npy")
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    rng = np.random.default_rng(SEED)
    folds, positions = assign_position_folds(meta, rng)
    return X_dense, y, meta, folds, positions


def fit_predict(X_tr, y_tr, X_te, cfg):
    model = xgb.XGBRegressor(
        tree_method="hist", n_jobs=-1, random_state=0, **cfg)
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def run_grid(args):
    X_dense, y, meta, folds, positions = load_data()
    results_path = OUT_DIR / "xgb_tune_grid.csv"
    done = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        done = set(zip(prev["fold"], prev["config_id"]))

    configs = []
    keys = list(GRID.keys())
    for combo in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, combo))
        cfg["subsample"] = 0.8
        cfg["colsample_bytree"] = 0.8
        configs.append(cfg)
    print(f"{len(configs)} candidate configs x {len(GRID_FOLDS)} folds "
          f"= {len(configs) * len(GRID_FOLDS)} fits")

    started = time.time()
    for k in GRID_FOLDS:
        train, test, n_mut = fold_masks(meta, positions, folds, k)
        scaler = StandardScaler().fit(X_dense[train])
        Xd = scaler.transform(X_dense).astype(np.float32)
        X_tr, y_tr = Xd[train], y[train]
        single = test & (n_mut == 1)
        X_te_single, y_te_single = Xd[single], y[single]
        if single.sum() == 0:
            continue

        for cid, cfg in enumerate(configs):
            if (k, cid) in done:
                continue
            if time.time() - started > TIME_BUDGET:
                print(f"time budget reached -- re-run to continue "
                      f"(next: fold {k}, config {cid})")
                return
            t0 = time.time()
            pred = fit_predict(X_tr, y_tr, X_te_single, cfg)
            rho = spearmanr(y_te_single, pred).statistic
            row = {"fold": k, "config_id": cid, "spearman_single": rho,
                   "n_single": int(single.sum()), "seconds": time.time() - t0,
                   **cfg}
            pd.DataFrame([row]).to_csv(
                results_path, mode="a", header=not results_path.exists(),
                index=False)
            print(f"fold {k} cfg {cid:3d} rho={rho:+.4f} "
                  f"({time.time() - t0:.1f}s) {cfg}", flush=True)

    print("\ngrid phase complete for folds", GRID_FOLDS)
    summarize_grid()


def summarize_grid():
    path = OUT_DIR / "xgb_tune_grid.csv"
    if not path.exists():
        print("no grid results yet")
        return
    df = pd.read_csv(path)
    n_folds_done = df["fold"].nunique()
    if n_folds_done < len(GRID_FOLDS):
        print(f"grid incomplete: {n_folds_done}/{len(GRID_FOLDS)} folds done")
        return
    agg = df.groupby("config_id").agg(
        mean_rho=("spearman_single", "mean"),
        max_depth=("max_depth", "first"), learning_rate=("learning_rate", "first"),
        n_estimators=("n_estimators", "first"),
        min_child_weight=("min_child_weight", "first")).sort_values(
        "mean_rho", ascending=False)
    print("\ntop 10 configs by mean single-mutant Spearman "
          f"(folds {GRID_FOLDS}):")
    print(agg.head(10).round(4).to_string())
    agg.to_csv(OUT_DIR / "xgb_tune_grid_summary.csv")


def run_eval(args):
    X_dense, y, meta, folds, positions = load_data()
    CV_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUT_DIR / "xgb_tune_grid_summary.csv"
    if not summary_path.exists():
        summarize_grid()
    agg = pd.read_csv(summary_path)
    top = agg.sort_values("mean_rho", ascending=False).head(3)
    candidates = {"default": DEFAULT_CFG}
    for _, r in top.iterrows():
        cfg = dict(max_depth=int(r["max_depth"]), learning_rate=r["learning_rate"],
                   n_estimators=int(r["n_estimators"]),
                   min_child_weight=int(r["min_child_weight"]),
                   subsample=0.8, colsample_bytree=0.8)
        candidates[f"cand_{int(r['config_id'])}"] = cfg
    print("evaluating candidates across all folds:")
    for name, cfg in candidates.items():
        print(f"  {name}: {cfg}")

    started = time.time()
    for k in range(N_FOLDS):
        train, test, n_mut = fold_masks(meta, positions, folds, k)
        scaler = StandardScaler().fit(X_dense[train])
        Xd = scaler.transform(X_dense).astype(np.float32)
        X_tr, y_tr, X_te = Xd[train], y[train], Xd[test]

        for name, cfg in candidates.items():
            path = CV_DIR / f"xgbtune_fold{k}_{name}.npz"
            if path.exists():
                continue
            if time.time() - started > TIME_BUDGET:
                print(f"time budget reached -- re-run to continue "
                      f"(next: fold {k}, {name})")
                return
            t0 = time.time()
            pred = fit_predict(X_tr, y_tr, X_te, cfg)
            np.savez_compressed(
                path, pred=pred, y_true=y[test], n_mut=n_mut[test],
                backbone=meta.loc[test, "Backbone"].to_numpy())
            print(f"fold {k} {name:10s} ({time.time() - t0:.1f}s)", flush=True)

    print("\nall eval folds complete -- run --phase analyze")


def run_analyze(args):
    files = sorted(CV_DIR.glob("xgbtune_fold*_*.npz"))
    names = sorted({f.stem.split("_", 2)[2] for f in files})
    pooled = {}
    for name in names:
        preds, ys, nmuts = [], [], []
        for k in range(N_FOLDS):
            p = CV_DIR / f"xgbtune_fold{k}_{name}.npz"
            if not p.exists():
                break
            d = np.load(p, allow_pickle=True)
            preds.append(d["pred"]); ys.append(d["y_true"]); nmuts.append(d["n_mut"])
        else:
            pooled[name] = (np.concatenate(preds), np.concatenate(ys), np.concatenate(nmuts))
    if len(pooled) < 2 or "default" not in pooled:
        print("not all folds/candidates complete yet")
        return

    print("\n=== pooled out-of-fold single-mutant Spearman ===")
    rows = []
    for name, (pred, ytrue, nmut) in pooled.items():
        single = nmut == 1
        rho = spearmanr(ytrue[single], pred[single]).statistic
        rows.append({"config": name, "n_single": int(single.sum()), "spearman_single": rho})
        print(f"{name:10s} n={single.sum():5d} rho={rho:+.4f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "xgb_tune_results.csv", index=False)

    rng = np.random.default_rng(0)
    default_pred, default_y, default_nmut = pooled["default"]
    single_mask = default_nmut == 1
    n = single_mask.sum()
    boot_rows = []
    for name, (pred, ytrue, nmut) in pooled.items():
        if name == "default":
            continue
        deltas = []
        idx_pool = np.flatnonzero(single_mask)
        for _ in range(4000):
            samp = rng.choice(idx_pool, size=len(idx_pool), replace=True)
            r1 = spearmanr(ytrue[samp], pred[samp]).statistic
            r0 = spearmanr(default_y[samp], default_pred[samp]).statistic
            deltas.append(r1 - r0)
        deltas = np.array(deltas)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        p_pos = float((deltas > 0).mean())
        boot_rows.append({"config": name, "delta_mean": deltas.mean(),
                          "ci_lo": lo, "ci_hi": hi, "p_gt_0": p_pos})
        print(f"{name} - default: delta={deltas.mean():+.4f} "
              f"CI[{lo:+.4f},{hi:+.4f}] P(>0)={p_pos:.3f}")
    pd.DataFrame(boot_rows).to_csv(OUT_DIR / "xgb_tune_bootstrap.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["grid", "eval", "analyze"], required=True)
    args = parser.parse_args()
    if args.phase == "grid":
        run_grid(args)
    elif args.phase == "eval":
        run_eval(args)
    else:
        run_analyze(args)


if __name__ == "__main__":
    main()
