"""
GFP variants dataset: step 11c — how wide should the ESM embedding block be?

Step 11b used 16 PCA components, which retain only 46% of the variance of the
1280-dim ESM-2 residue embeddings, and still gained +0.040 rho. That number was
picked to be safe, not because it was optimal, so this sweeps it.

The tension: more components carry more structural context, but the block sits
next to 66 dense features on 141k rows, and the project has twice been burned
by apparent gains that were sampling noise (findings 5). So every setting is
run on the full 5-fold position CV with the identical folds, and the winner is
confirmed by a paired bootstrap against the 16-dim version rather than by its
point estimate.

Run with: python esm_dims_sweep.py
Output: output/esm_dims_sweep.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import importlib, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
_evo_benchmark = importlib.import_module("18_evo_benchmark")
assign_position_folds = _evo_benchmark.assign_position_folds
fold_masks = _evo_benchmark.fold_masks
N_FOLDS = _evo_benchmark.N_FOLDS
SEED = _evo_benchmark.SEED
XGB_CFG = _evo_benchmark.XGB_CFG
esm_embed = importlib.import_module("24_esm_embed")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"
CV_DIR = OUT_DIR / "extrap_cv"

DIMS = [8, 16, 32, 64]
HF_MODEL = "facebook/esm2_t33_650M_UR50D"
N_BOOT = 4000


def features_for(dims, wt_sequences, meta):
    f = FEAT_DIR / f"X_esmemb_d{dims}.npy"
    if f.exists():
        return np.load(f)
    emb = esm_embed.wt_embeddings(wt_sequences, HF_MODEL, OUT_DIR / "esm_cache")
    reduced = esm_embed.reduce_embeddings(emb, dims)
    X, _names = esm_embed.build_features(meta, reduced, wt_sequences, dims)
    np.save(f, X)
    return X


def main():
    import json
    wt_sequences = json.loads((OUT_DIR / "backbone_wt.json").read_text())
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    y = np.load(FEAT_DIR / "y.npy")
    Xd = np.load(FEAT_DIR / "X_dense.npy")
    Xe = np.load(FEAT_DIR / "X_evo.npy")

    rng = np.random.default_rng(SEED)
    folds, positions = assign_position_folds(meta, rng)

    preds, rows = {}, []
    for dims in DIMS:
        Xb = features_for(dims, wt_sequences, meta)
        X_all = np.hstack([Xd, Xe, Xb])
        cached = CV_DIR / f"dims{dims}_pred.npz"
        if cached.exists():
            d = np.load(cached, allow_pickle=True)
            preds[dims] = (d["pred"], d["y_true"], d["n_mut"], d["backbone"])
        else:
            P, Y, M, B = [], [], [], []
            t0 = time.time()
            for k in range(N_FOLDS):
                train, test, n_mut = fold_masks(meta, positions, folds, k)
                sc = StandardScaler().fit(X_all[train])
                X = sc.transform(X_all).astype(np.float32)
                m = xgb.XGBRegressor(tree_method="hist", n_jobs=-1,
                                     random_state=0, **XGB_CFG)
                m.fit(X[train], y[train])
                P.append(m.predict(X[test])); Y.append(y[test])
                M.append(n_mut[test])
                B.append(meta.loc[test, "Backbone"].to_numpy())
            P, Y, M, B = map(np.concatenate, (P, Y, M, B))
            np.savez_compressed(cached, pred=P, y_true=Y, n_mut=M, backbone=B)
            preds[dims] = (P, Y, M, B)
            print(f"dims {dims:3d} ({X_all.shape[1]} features) "
                  f"({time.time() - t0:.0f}s)", flush=True)

        P, Y, M, B = preds[dims]
        single = M == 1
        rows.append({"dims": dims, "n_features": X_all.shape[1],
                     "rho_all": round(spearmanr(P, Y).statistic, 4),
                     "rho_single": round(spearmanr(P[single], Y[single]).statistic, 4)})

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    # paired bootstrap of every setting against 16 dims
    P16, Y, M, _B = preds[16]
    single = np.where(M == 1)[0]
    rngb = np.random.default_rng(0)
    boot = []
    for dims in DIMS:
        if dims == 16:
            continue
        P = preds[dims][0]
        diffs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            s = rngb.choice(single, len(single), replace=True)
            diffs[b] = (spearmanr(P[s], Y[s]).statistic
                        - spearmanr(P16[s], Y[s]).statistic)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        boot.append({"dims": dims, "vs": 16, "mean_diff": round(float(diffs.mean()), 4),
                     "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                     "p_gt_0": float((diffs > 0).mean())})
        print(f"dims {dims:3d} − 16: {diffs.mean():+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}], P(>0) = {(diffs > 0).mean():.3f}")

    df.to_csv(OUT_DIR / "esm_dims_sweep.csv", index=False)
    pd.DataFrame(boot).to_csv(OUT_DIR / "esm_dims_bootstrap.csv", index=False)


if __name__ == "__main__":
    main()
