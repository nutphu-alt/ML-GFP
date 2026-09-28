"""
GFP variants dataset: step 11 — does ESM-2 beat the cross-homolog features?

Step 9's argument for ESM-2 was empirical: three homologous sequences were
worth +0.044 rho at never-assayed positions, so a model distilled from
millions should be worth more, and it would have no ppluGFP2 blind spot
(ppluGFP2 is the one backbone the evo features HURT, being the phylogenetic
outlier at 18-25% identity).

That was untestable until now — dl.fbaipublicfiles.com and download.pytorch.org
are blocked, but huggingface.co is reachable from the cloud container, so the
weights arrive by a different route (see esm_scores.py --hf-model).

Four arms on the settled 5-fold position CV (SEED=7, folds imported from
evo_benchmark so they are identical to steps 5, 8, 9 and 10a):

    dense           66   the step-8 baseline
    dense+evo       77   the step-9 winner
    dense+esm       70   ESM-2 instead of the homolog features
    dense+evo+esm   81   both

All four are fitted here on identical folds rather than reusing older
predictions, so every comparison is exactly paired.

Reports the per-backbone breakdown too, because the ppluGFP2 question is the
specific claim step 9 made about what a PLM would fix.

Resume-safe: one npz per (arm, fold).

Run with: python esm_benchmark.py [--esm-suffix _35M]
Output: output/extrap_cv/esm_{arm}_fold{k}.npz, output/esm_benchmark.csv,
        output/esm_bootstrap.csv
"""

import argparse
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

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"
CV_DIR = OUT_DIR / "extrap_cv"
N_BOOT = 4000
TIME_BUDGET = 480


def arms(suffix, with_emb=False):
    Xd = np.load(FEAT_DIR / "X_dense.npy")
    Xe = np.load(FEAT_DIR / "X_evo.npy")
    Xm = np.load(FEAT_DIR / f"X_esm{suffix}.npy")
    out = {
        "dense": Xd,
        "dense+evo": np.hstack([Xd, Xe]),
        "dense+esm": np.hstack([Xd, Xm]),
        "dense+evo+esm": np.hstack([Xd, Xe, Xm]),
    }
    if with_emb:
        Xb = np.load(FEAT_DIR / f"X_esmemb{suffix}.npy")
        out["dense+emb"] = np.hstack([Xd, Xb])
        out["dense+evo+emb"] = np.hstack([Xd, Xe, Xb])
        out["dense+evo+esm+emb"] = np.hstack([Xd, Xe, Xm, Xb])
    return out


def run(suffix, with_emb=False):
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    blocks = arms(suffix, with_emb)
    rng = np.random.default_rng(SEED)
    folds, positions = assign_position_folds(meta, rng)

    started = time.time()
    for name, X_all in blocks.items():
        for k in range(N_FOLDS):
            path = CV_DIR / f"esm{suffix}_{name}_fold{k}.npz"
            if path.exists():
                continue
            if time.time() - started > TIME_BUDGET:
                print(f"time budget reached — re-run to continue "
                      f"(next: {name} fold {k})")
                return False
            train, test, n_mut = fold_masks(meta, positions, folds, k)
            scaler = StandardScaler().fit(X_all[train])
            X = scaler.transform(X_all).astype(np.float32)
            m = xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0,
                                 **XGB_CFG)
            m.fit(X[train], y[train])
            np.savez_compressed(
                path, pred=m.predict(X[test]), y_true=y[test],
                n_mut=n_mut[test],
                backbone=meta.loc[test, "Backbone"].to_numpy())
            print(f"{name:14s} fold {k} test={test.sum():5d} "
                  f"({time.time() - started:.0f}s)", flush=True)
    return True


def pooled(suffix, name):
    out = {k: [] for k in ["pred", "y_true", "n_mut", "backbone"]}
    for k in range(N_FOLDS):
        d = np.load(CV_DIR / f"esm{suffix}_{name}_fold{k}.npz", allow_pickle=True)
        for key in out:
            out[key].append(d[key])
    return {k: np.concatenate(v) for k, v in out.items()}


def analyse(suffix, with_emb=False):
    data = {n: pooled(suffix, n) for n in arms(suffix, with_emb)}
    ref = data["dense"]
    for n, d in data.items():
        assert np.allclose(d["y_true"], ref["y_true"]), f"{n} rows disagree"

    y, nmut, bb = ref["y_true"], ref["n_mut"], ref["backbone"]
    single = nmut == 1
    rows = []
    for n, d in data.items():
        p = d["pred"]
        rows.append({"arm": n, "subset": "all novel", "n": len(y),
                     "spearman": spearmanr(p, y).statistic})
        rows.append({"arm": n, "subset": "single mutants", "n": int(single.sum()),
                     "spearman": spearmanr(p[single], y[single]).statistic})
        for b in sorted(set(bb)):
            m = single & (bb == b)
            rows.append({"arm": n, "subset": f"single — {b}", "n": int(m.sum()),
                         "spearman": spearmanr(p[m], y[m]).statistic})
    df = pd.DataFrame(rows)
    df["spearman"] = df["spearman"].round(4)

    # paired bootstrap on single mutants, against the step-9 winner
    rngb = np.random.default_rng(0)
    idx_pool = np.where(single)[0]
    boot = []
    base = data["dense+evo"]["pred"]
    cands = [n for n in data if n not in ("dense", "dense+evo")]
    for n in cands:
        cand = data[n]["pred"]
        diffs = np.empty(N_BOOT)
        for b in range(N_BOOT):
            s = rngb.choice(idx_pool, len(idx_pool), replace=True)
            diffs[b] = (spearmanr(cand[s], y[s]).statistic
                        - spearmanr(base[s], y[s]).statistic)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        boot.append({"comparison": f"{n} − dense+evo, single mutants",
                     "mean_diff": round(float(diffs.mean()), 4),
                     "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                     "p_gt_0": float((diffs > 0).mean())})
        print(f"{n:16s} − dense+evo: {diffs.mean():+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}], P(>0) = {(diffs > 0).mean():.3f}")
    return df, pd.DataFrame(boot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm-suffix", default="_35M")
    ap.add_argument("--with-embeddings", action="store_true")
    args = ap.parse_args()
    CV_DIR.mkdir(parents=True, exist_ok=True)
    if not run(args.esm_suffix, args.with_embeddings):
        return
    df, boot = analyse(args.esm_suffix, args.with_embeddings)
    df.to_csv(OUT_DIR / f"esm_benchmark{args.esm_suffix}.csv", index=False)
    boot.to_csv(OUT_DIR / f"esm_bootstrap{args.esm_suffix}.csv", index=False)
    print("\n" + df.pivot_table(index="subset", columns="arm",
                                values="spearman").round(4).to_string())


if __name__ == "__main__":
    main()
