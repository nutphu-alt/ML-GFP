"""
GFP variants dataset: step 12 — re-optimise the DESIGN oracle.

Step 10 built the recombination design oracle on a configuration inherited
from step 8. That configuration was selected for a different problem:
extrapolation to never-assayed positions, on dense features only. Step 8's
negative result ("tuning is exhausted") was established in THAT regime and
does not transfer to this one — recombination is interpolation, over a
6,636-feature block including the sparse indicators. It has never been tuned.

Two things are optimised here:

  features   does the step-11 ESM embedding block help the design regime too?
             It lifted never-assayed positions 0.479 -> 0.519 and LOBO
             0.520 -> 0.553, but the design regime is a different problem and
             already has the sparse block doing the site-specific work.

  config     an XGBoost grid, selected on the objective that actually matters
             for a bench panel: the mean measured brightness of the top 20
             picks, not overall Spearman.

PROTOCOL, following the project's standing convention and finding 5 (small-
sample checks overstate differences, twice burned):

  - the grid is scored on the random split's VALIDATION rows, never test
  - exactly one configuration is then confirmed on the TEST rows
  - the confirmation is a paired bootstrap against the step-10 incumbent
  - the selection objective is mean-y@20, a continuous quantity, because
    precision@20 at a 0.5% base rate is far too noisy to select on

Run with: python design_optimize.py
Output: output/design_opt_grid.csv, output/design_opt_confirm.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

import importlib, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
dv = importlib.import_module("21_design_variants")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"

N_BOOT = 4000
INCUMBENT = dict(n_estimators=400, max_depth=8, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8)

GRID = [
    dict(n_estimators=400,  max_depth=6,  learning_rate=0.1,  subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=400,  max_depth=8,  learning_rate=0.1,  subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=400,  max_depth=10, learning_rate=0.1,  subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=400,  max_depth=12, learning_rate=0.1,  subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=800,  max_depth=8,  learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=800,  max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(n_estimators=800,  max_depth=12, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5),
    dict(n_estimators=1200, max_depth=10, learning_rate=0.03, subsample=0.8, colsample_bytree=0.5),
    dict(n_estimators=400,  max_depth=10, learning_rate=0.1,  subsample=0.6, colsample_bytree=0.5),
    dict(n_estimators=800,  max_depth=14, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5),
]


def load_all():
    Xd = np.load(FEAT_DIR / "X_dense.npy")
    Xe = np.load(FEAT_DIR / "X_evo.npy")
    Xs = sp.load_npz(FEAT_DIR / "X_sparse.npz").tocsr()
    Xb = np.load(FEAT_DIR / "X_esmemb_650M.npy")
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    return Xd, Xe, Xs, Xb, y, meta


def regime_masks(meta, y):
    """Design-regime rows inside validation and inside test, separately."""
    muts = [dv.parse_name(n) for n in meta["Variant name"]]
    nmut = np.array([len(m) for m in muts])
    sr = meta["split_random"].to_numpy()
    tr = sr == "train"
    out = {}
    for name in ("val", "test"):
        keep, _seen = dv.design_regime_mask(meta, muts, nmut, tr, sr == name)
        out[name] = keep
    return tr, out


def assemble(Xd, Xe, Xs, Xb, tr, with_emb):
    dn = np.hstack([Xd, Xe, Xb]) if with_emb else np.hstack([Xd, Xe])
    sc = StandardScaler().fit(dn[tr])
    return sp.hstack([sp.csr_matrix(sc.transform(dn).astype(np.float32)),
                      Xs]).tocsr()


def mean_y_at_k(pred, y, k=20):
    return float(y[np.argsort(-pred)[:k]].mean())


def precision_at_k(pred, y, thr, k=20):
    return float((y[np.argsort(-pred)[:k]] >= thr).mean())


def main():
    Xd, Xe, Xs, Xb, y, meta = load_all()
    tr, masks = regime_masks(meta, y)
    val, test = masks["val"], masks["test"]
    print(f"design-regime rows — validation {val.sum()}, test {test.sum()}")

    cache = {}
    rows = []
    for with_emb in (False, True):
        X = assemble(Xd, Xe, Xs, Xb, tr, with_emb)
        cache[with_emb] = X
        tag = "dense+evo+sparse+emb" if with_emb else "dense+evo+sparse"
        for i, cfg in enumerate(GRID):
            t0 = time.time()
            m = xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0,
                                 **cfg)
            m.fit(X[tr], y[tr])
            p = m.predict(X[val])
            rows.append({
                "features": tag, "n_features": X.shape[1], "config_id": i, **cfg,
                "val_mean_y_at20": round(mean_y_at_k(p, y[val]), 4),
                "val_prec20_1.2": round(precision_at_k(p, y[val], 1.2), 3),
                "val_prec20_1.5": round(precision_at_k(p, y[val], 1.5), 3),
                "val_rho": round(float(spearmanr(p, y[val]).statistic), 4),
                "secs": round(time.time() - t0, 1),
            })
            print(f"  {tag:22s} cfg{i:2d} d{cfg['max_depth']:<3d} "
                  f"n{cfg['n_estimators']:<5d} "
                  f"mean_y@20 {rows[-1]['val_mean_y_at20']:.4f} "
                  f"({rows[-1]['secs']:.0f}s)", flush=True)

    grid = pd.DataFrame(rows)
    grid.to_csv(OUT_DIR / "design_opt_grid.csv", index=False)
    best = grid.loc[grid["val_mean_y_at20"].idxmax()]
    print(f"\nselected on VALIDATION: {best['features']}, config "
          f"{int(best['config_id'])} — mean_y@20 {best['val_mean_y_at20']}")

    # ---- confirm exactly one winner on test, paired against the incumbent ---
    best_cfg = {k: best[k] for k in
                ("n_estimators", "max_depth", "learning_rate",
                 "subsample", "colsample_bytree")}
    best_cfg["n_estimators"] = int(best_cfg["n_estimators"])
    best_cfg["max_depth"] = int(best_cfg["max_depth"])
    best_emb = best["features"].endswith("emb")

    Xw = cache[best_emb]
    Xi = cache[False]
    pw = xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0,
                          **best_cfg).fit(Xw[tr], y[tr]).predict(Xw[test])
    pi = xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0,
                          **INCUMBENT).fit(Xi[tr], y[tr]).predict(Xi[test])
    yt = y[test]

    rng = np.random.default_rng(0)
    n = len(yt)
    d_mean = np.empty(N_BOOT); d_p15 = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        d_mean[b] = mean_y_at_k(pw[s], yt[s]) - mean_y_at_k(pi[s], yt[s])
        d_p15[b] = (precision_at_k(pw[s], yt[s], 1.5)
                    - precision_at_k(pi[s], yt[s], 1.5))

    out = []
    for label, arr in [("mean_y@20", d_mean), ("precision@20 at y>=1.5", d_p15)]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out.append({"metric": label, "mean_diff": round(float(arr.mean()), 4),
                    "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                    "p_gt_0": float((arr > 0).mean())})
        print(f"  tuned − incumbent, {label}: {arr.mean():+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}], P(>0) = {(arr > 0).mean():.3f}")

    conf = pd.DataFrame(out)
    conf["winner_features"] = best["features"]
    conf["winner_config"] = str(best_cfg)
    for label, p in [("incumbent", pi), ("tuned", pw)]:
        print(f"  {label:10s} TEST mean_y@20 {mean_y_at_k(p, yt):.4f}  "
              f"prec@20 y>=1.2 {precision_at_k(p, yt, 1.2):.2f}  "
              f"y>=1.5 {precision_at_k(p, yt, 1.5):.2f}  "
              f"rho {spearmanr(p, yt).statistic:.4f}")
        conf[f"{label}_test_mean_y_at20"] = round(mean_y_at_k(p, yt), 4)
        conf[f"{label}_test_prec20_1.5"] = round(precision_at_k(p, yt, 1.5), 3)
    conf.to_csv(OUT_DIR / "design_opt_confirm.csv", index=False)


if __name__ == "__main__":
    main()
