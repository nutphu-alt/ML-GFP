"""
GFP variants dataset: step 13 — how much of the design result is near-neighbours?

Step 10's design-regime benchmark evaluates novel COMBINATIONS of characterised
substitutions on the random split. That is the correct regime for recombination
design, but it is the optimistic end of this project's evaluations: DMS variants
differ from their parent by 2-3 mutations, so a held-out 3-mutant may sit one
substitution away from a training variant. A model can then score well by
near-duplicate lookup rather than by combining evidence.

This measures that directly. For every design-regime test row, ask whether any
TRAINING row of the same backbone shares all but one of its substitutions. Rows
that have such a neighbour are the easy case; rows that do not are genuine
combination inference.

    has_neighbour      >=1 training variant sharing n-1 of its substitutions
    no_neighbour       none — the model must actually combine

Reporting both, with the paired gap bootstrapped, replaces the single
optimistic number with an honest range.

Run with: python design_stress_test.py
Output: output/design_stress_test.csv, output/design_stress_bootstrap.csv
"""

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
CFG = dict(n_estimators=400, max_depth=8, learning_rate=0.1,
           subsample=0.8, colsample_bytree=0.8)


def neighbour_flags(meta, muts, nmut, train_mask, eval_mask):
    """True where a training variant shares all but one substitution."""
    bb = meta["Backbone"].to_numpy()
    # index training variants by every (n-1)-subset of their substitution set
    index = {}
    for i in np.where(train_mask)[0]:
        subs = frozenset((p, m) for _w, p, m in muts[i])
        if not subs:
            continue
        for drop in subs:
            index.setdefault((bb[i], subs - {drop}), 0)
            index[(bb[i], subs - {drop})] += 1
        index.setdefault((bb[i], subs), 0)
        index[(bb[i], subs)] += 1

    flags = np.zeros(len(meta), bool)
    for i in np.where(eval_mask)[0]:
        subs = frozenset((p, m) for _w, p, m in muts[i])
        # an (n-1)-subset of the eval variant matching a training variant's
        # full set, or the eval set itself being a training (n-1)-subset
        hit = (bb[i], subs) in index
        if not hit:
            for drop in subs:
                if (bb[i], subs - {drop}) in index:
                    hit = True
                    break
        flags[i] = hit
    return flags


def prec_at_k(pred, y, thr, k):
    return float((y[np.argsort(-pred)[:k]] >= thr).mean())


def main():
    Xd = np.load(FEAT_DIR / "X_dense.npy")
    Xe = np.load(FEAT_DIR / "X_evo.npy")
    Xs = sp.load_npz(FEAT_DIR / "X_sparse.npz").tocsr()
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")

    muts = [dv.parse_name(n) for n in meta["Variant name"]]
    nmut = np.array([len(m) for m in muts])
    sr = meta["split_random"].to_numpy()
    tr, te = sr == "train", sr == "test"
    keep, _ = dv.design_regime_mask(meta, muts, nmut, tr, te)

    nb = neighbour_flags(meta, muts, nmut, tr, keep)
    easy = keep & nb
    hard = keep & ~nb
    print(f"design-regime test rows: {keep.sum()}")
    print(f"  with a near neighbour in training: {easy.sum()} "
          f"({easy.sum()/keep.sum():.1%})")
    print(f"  without one (true combination):    {hard.sum()} "
          f"({hard.sum()/keep.sum():.1%})")

    dn = np.hstack([Xd, Xe])
    sc = StandardScaler().fit(dn[tr])
    X = sp.hstack([sp.csr_matrix(sc.transform(dn).astype(np.float32)), Xs]).tocsr()
    P = np.mean([xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=s,
                                  **CFG).fit(X[tr], y[tr]).predict(X[keep])
                 for s in range(3)], axis=0)

    sub = {"all design-regime": np.ones(int(keep.sum()), bool),
           "has_neighbour": nb[keep],
           "no_neighbour": ~nb[keep]}
    yk = y[keep]
    rows = []
    for label, m in sub.items():
        if m.sum() < 50:
            continue
        p, yy = P[m], yk[m]
        for thr in (1.0, 1.2, 1.5):
            for k in (10, 20, 50):
                rows.append({"subset": label, "n": int(m.sum()), "threshold": thr,
                             "k": k, "precision": round(prec_at_k(p, yy, thr, k), 3),
                             "baseline": round(float((yy >= thr).mean()), 4),
                             "spearman": round(float(spearmanr(p, yy).statistic), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "design_stress_test.csv", index=False)
    print("\n" + df[df["k"] == 20].pivot_table(
        index="threshold", columns="subset", values="precision").to_string())

    # bootstrap the gap on the metric the panel is built on
    rng = np.random.default_rng(0)
    ih, ie = np.where(~nb[keep])[0], np.where(nb[keep])[0]
    out = []
    for thr in (1.2, 1.5):
        d = np.empty(N_BOOT)
        for b in range(N_BOOT):
            a = rng.choice(ie, len(ie), replace=True)
            c = rng.choice(ih, len(ih), replace=True)
            d[b] = prec_at_k(P[c], yk[c], thr, 20) - prec_at_k(P[a], yk[a], thr, 20)
        lo, hi = np.percentile(d, [2.5, 97.5])
        out.append({"metric": f"precision@20 at y>={thr}",
                    "no_neighbour_minus_has_neighbour": round(float(d.mean()), 4),
                    "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)})
        print(f"  no_neighbour − has_neighbour, precision@20 y>={thr}: "
              f"{d.mean():+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}]")
    pd.DataFrame(out).to_csv(OUT_DIR / "design_stress_bootstrap.csv", index=False)


if __name__ == "__main__":
    main()
