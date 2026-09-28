"""
GFP variants dataset: step 10 — design / optimisation.

This is the second half of the project's objective: use the predictor to
propose sequences that should be bright. It is NOT a straightforward
"maximise the step-9 model" loop, and the reason is the main finding of this
step.

WHY THE OBVIOUS DESIGN LOOP DOES NOT WORK
-----------------------------------------
Checking the step-9 out-of-fold predictions before trusting them showed that
the ~0.45 Spearman at never-seen positions is almost entirely dead-vs-alive
separation, not brightness ranking:

    overall rho 0.455  (AUC alive 0.778)      y>=0.5:  rho  0.247
    y>=0.8:     rho 0.130                     y>=1.0:  rho -0.146
    variants with y>=1.2 sit at median predicted rank 6452 / 16017
    top-20 of the ranked list: nothing above 1.2x WT (enrichment < 1)

Step 10a (`two_stage_benchmark.py`) confirmed this is not fixable by splitting
the problem: a dedicated classifier moves AUC 0.778 -> 0.791, a regressor fit
on functional rows only moves Spearman 0.247 -> 0.262, and precision for
y>=1.2 stays at baseline. So at NEVER-ASSAYED positions the model is a
foldability filter, and maximising it is optimising noise.

But this is not an assay ceiling. In the interpolation regime the bright end
is sharply predictable (random split, top-20 precision for y>=1.2 = 1.00
against a 3.0% baseline). The signal is real and learnable; what fails is
transfer to novel sites.

WHAT THIS STEP THEREFORE DOES
-----------------------------
Design by RECOMBINATION: propose novel combinations of substitutions that have
each already been individually characterised in that backbone's library. That
places the design squarely in the regime where the model works, and it is
directly measurable — the `benchmark` phase holds out exactly that population
(>=2 mutations, every substitution seen in training, combination unseen) and
reports the precision an experimentalist should expect.

A second reversal falls out of it. The sparse (backbone, position, mutant)
block was retired in steps 4/7/8 as dead weight, correctly, FOR EXTRAPOLATION.
For recombination it is essential — it is precisely the memorised per-
substitution knowledge that recombination exploits. It lifts top-10 precision
at y>=1.5 from 0.10 to 0.60-0.80. The design oracle therefore uses
dense + evo + sparse (6,636 features), unlike every other model in this
project.

Phases
------
  validate   re-derive features for real training rows, assert they reproduce
             the stored matrices exactly (guards against feature-space drift)
  benchmark  design-regime precision, sparse vs dense, with a paired bootstrap
  oracle     fit the seed-ensemble on all 141,144 DMS rows
  design     beam search over characterised substitutions -> ranked shortlist
  report     figure

Resume-safe: every phase caches its output.

Run with: python design_variants.py [--backbone avGFP] [--phase all]
Output: output/design_regime_benchmark.csv, output/design_oracle.joblib,
        output/design_singles.csv, output/design_shortlist.csv,
        output/design.png
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

import importlib, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
bf = importlib.import_module("03_build_features")
ef = importlib.import_module("17_evo_features")

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"

XGB_CFG = dict(n_estimators=400, max_depth=8, learning_rate=0.1,
               subsample=0.8, colsample_bytree=0.8)
N_SEEDS = 5
N_BOOT = 4000
BACKBONES = ["amacGFP", "avGFP", "cgreGFP", "ppluGFP2"]

# Chromophore tripeptide (Thr65-Tyr66-Gly67) plus the Arg96/Glu222 pair that
# catalyses its maturation, in avGFP numbering, mapped to other backbones
# through the step-9a alignment. Mutating these is not design, it is breaking
# the chromophore.
AVGFP_FORBIDDEN = [65, 66, 67, 96, 222]

BEAM_WIDTH = 150
EXTEND_POOL = 80
MAX_DEPTH = 5

# Components must be individually measured and individually non-deleterious.
# This is not a heuristic — it is the single most important constraint in this
# step, and it is set by measurement. Among avGFP multi-mutants whose every
# component was assayed alone:
#
#   every component >=1.0x WT   2 mut  mean y 1.078 | 3 mut 1.103 | 4 mut 1.074
#   every component >=0.9x WT   2 mut  mean y 0.991 | 3 mut 0.982 | ... 6 mut 0.914
#   unfiltered                  2 mut  mean y 0.701 | 3 mut 0.545 | ... 6 mut 0.118
#
# So the well-known collapse of brightness with mutation count is caused by
# accumulating deleterious mutations, NOT by stacking as such. Conditioned on
# good components, brightness is roughly preserved to 5-6 mutations. An
# unconstrained beam search ignores this, stacks whatever the model scores
# highly, and produces deep designs whose predictions rise with depth while
# their empirical base rate falls ~28x — the search exploiting model optimism.
MIN_COMPONENT = 0.95


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_refs():
    return (json.loads((OUT_DIR / "backbone_wt.json").read_text()),
            json.loads((OUT_DIR / "backbone_alignment.json").read_text()))


def load_training():
    Xd = np.load(FEAT_DIR / "X_dense.npy")
    Xe = np.load(FEAT_DIR / "X_evo.npy")
    Xs = sp.load_npz(FEAT_DIR / "X_sparse.npz").tocsr()
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    sparse_names = (FEAT_DIR / "sparse_feature_names.txt").read_text().split()
    return Xd, Xe, Xs, y, meta, sparse_names


def parse_name(name):
    return [(w, int(p), m) for w, p, m in bf.MUTATION_RE.findall(str(name))]


def variant_name(muts):
    return ":".join(f"{w}{p}{m}" for w, p, m in sorted(muts, key=lambda t: t[1]))


def apply_mutations(seq, muts):
    s = list(seq)
    for _w, p, m in muts:
        s[p - 1] = m
    return "".join(s)


# --------------------------------------------------------------------------
# featurisation - delegates to the project's own feature code
# --------------------------------------------------------------------------

def featurise_dense_evo(records, wt_sequences, alignment):
    """records: [(backbone, [(wt,pos,mut),...])] -> (n, 77) in training layout.

    Four dummy rows (one per backbone) are appended so build_dense_block's
    `sorted(df["Backbone"].unique())` lays out the one-hot columns exactly as
    training did, then dropped.
    """
    rows, muts_list = [], []
    for backbone, muts in records:
        rows.append({"Backbone": backbone,
                     "Protein sequence": apply_mutations(wt_sequences[backbone], muts),
                     "Variant name": variant_name(muts)})
        muts_list.append(muts)
    for b in BACKBONES:
        rows.append({"Backbone": b, "Protein sequence": wt_sequences[b],
                     "Variant name": ""})
        muts_list.append([])

    df = pd.DataFrame(rows)
    Xd, dense_names = bf.build_dense_block(df, pd.Series(muts_list))
    with contextlib.redirect_stdout(io.StringIO()):
        Xe = ef.build_features(df, alignment, wt_sequences)
    n = len(records)
    return np.hstack([Xd[:n], Xe[:n]]).astype(np.float32), dense_names + ef.FEATURE_NAMES


def sparse_lookup(sparse_names):
    return {n: i for i, n in enumerate(sparse_names)}


def featurise_sparse(records, lookup, n_cols):
    rows, cols = [], []
    missing = 0
    for i, (backbone, muts) in enumerate(records):
        for _w, p, m in muts:
            c = lookup.get(f"{backbone}:{p}{m}")
            if c is None:
                missing += 1
                continue
            rows.append(i); cols.append(c)
    X = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(len(records), n_cols), dtype=np.float32).tocsr()
    return X, missing


class Oracle:
    """dense+evo (scaled) hstacked with the raw sparse block, seed-ensembled."""

    def __init__(self, models, scaler, lookup, n_sparse):
        self.models, self.scaler = models, scaler
        self.lookup, self.n_sparse = lookup, n_sparse

    def featurise(self, records, wt_sequences, alignment):
        """Returns (X for the model, raw unscaled dense+evo, names, missing).

        The raw block is returned alongside because the scaled copy is
        z-scores — reporting cons_min from it would be meaningless.
        """
        raw, names = featurise_dense_evo(records, wt_sequences, alignment)
        Xde = self.scaler.transform(raw).astype(np.float32)
        Xs, missing = featurise_sparse(records, self.lookup, self.n_sparse)
        return sp.hstack([sp.csr_matrix(Xde), Xs]).tocsr(), raw, names, missing

    def predict(self, X):
        P = np.column_stack([m.predict(X) for m in self.models])
        return P.mean(axis=1), P.std(axis=1)


def fit_ensemble(X, y, n_seeds=N_SEEDS, tag=""):
    models = []
    for s in range(n_seeds):
        t0 = time.time()
        m = xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=s,
                             **XGB_CFG)
        m.fit(X, y)
        models.append(m)
        print(f"  {tag}seed {s} ({time.time() - t0:.0f}s)", flush=True)
    return models


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------

def phase_validate(wt_sequences, alignment):
    Xd, Xe, Xs, _y, meta, sparse_names = load_training()
    ref = np.hstack([Xd, Xe])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(meta), size=4000, replace=False)

    records = [(meta.at[i, "Backbone"], parse_name(meta.at[i, "Variant name"]))
               for i in idx]
    new, names = featurise_dense_evo(records, wt_sequences, alignment)
    d = np.abs(new - ref[idx])
    bad = [n for n, w in zip(names, d.max(axis=0)) if w > 1e-3]
    print(f"dense+evo reproduction: {int((d.max(axis=1) <= 1e-3).sum())}/{len(idx)} "
          f"rows exact, {len(bad)} columns disagree")

    Xs_new, missing = featurise_sparse(records, sparse_lookup(sparse_names),
                                       Xs.shape[1])
    sdiff = abs(Xs_new - Xs[idx]).max()
    print(f"sparse reproduction: max abs diff {sdiff:.4g}, "
          f"{missing} substitutions absent from the training vocabulary")
    return len(bad) == 0 and sdiff <= 1e-6


def design_regime_mask(meta, muts, nmut, train_mask, test_mask):
    """Test rows that ARE the design use case: >=2 mutations, every
    substitution individually characterised in training, combination unseen."""
    bbone = meta["Backbone"].to_numpy()
    seen = {}
    for i in np.where(train_mask)[0]:
        seen.setdefault(bbone[i], set()).update((p, m) for _w, p, m in muts[i])
    keep = np.zeros(len(meta), bool)
    for i in np.where(test_mask)[0]:
        S = seen.get(bbone[i], set())
        if nmut[i] >= 2 and all((p, m) in S for _w, p, m in muts[i]):
            keep[i] = True
    return keep, seen


def phase_benchmark():
    """Design-regime precision, sparse vs dense, with a paired bootstrap."""
    Xd, Xe, Xs, y, meta, _sn = load_training()
    muts = [parse_name(n) for n in meta["Variant name"]]
    nmut = np.array([len(m) for m in muts])
    sr = meta["split_random"].to_numpy()
    tr, te = sr == "train", sr == "test"
    keep, _seen = design_regime_mask(meta, muts, nmut, tr, te)
    print(f"design-regime evaluation rows: {keep.sum()} "
          f"(mean {nmut[keep].mean():.2f} mutations)")

    dn = np.hstack([Xd, Xe])
    scaler = StandardScaler().fit(dn[tr])
    Xdense = scaler.transform(dn).astype(np.float32)
    Xall = sp.hstack([sp.csr_matrix(Xdense), Xs]).tocsr()

    preds = {}
    for label, X in [("dense+evo (77)", Xdense),
                     ("dense+evo+sparse (%d)" % Xall.shape[1], Xall)]:
        P = [xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=s,
                              **XGB_CFG).fit(X[tr], y[tr]).predict(X[keep])
             for s in range(3)]
        preds[label] = np.mean(P, axis=0)
        print(f"  fitted {label}", flush=True)

    yt = y[keep]
    rows = []
    for label, p in preds.items():
        o = np.argsort(-p)
        for thr in [1.0, 1.2, 1.5]:
            for k in [10, 20, 50, 100]:
                rows.append({
                    "model": label, "threshold": thr, "k": k,
                    "precision": float((yt[o[:k]] >= thr).mean()),
                    "baseline": float((yt >= thr).mean()),
                    "enrichment": float((yt[o[:k]] >= thr).mean() /
                                        max((yt >= thr).mean(), 1e-9)),
                    "spearman_all": float(spearmanr(p, yt).statistic),
                })
    df = pd.DataFrame(rows)

    # paired bootstrap on the headline claim: sparse beats dense at y>=1.5, k=20
    labels = list(preds)
    rng = np.random.default_rng(0)
    diffs = np.empty(N_BOOT)
    n = len(yt)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = yt[idx]
        pa, pb = preds[labels[0]][idx], preds[labels[1]][idx]
        oa, ob = np.argsort(-pa)[:20], np.argsort(-pb)[:20]
        diffs[b] = (yb[ob] >= 1.5).mean() - (yb[oa] >= 1.5).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\npaired bootstrap, precision@20 at y>=1.5, sparse - dense: "
          f"{diffs.mean():+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}], "
          f"P(>0) = {(diffs > 0).mean():.3f}")
    boot = pd.DataFrame([{"comparison": "sparse - dense, precision@20, y>=1.5",
                          "mean_diff": diffs.mean(), "ci_lo": lo, "ci_hi": hi,
                          "p_gt_0": float((diffs > 0).mean())}])
    return df, boot


def forbidden_positions(backbone, alignment):
    p2c = alignment["pos_to_col"]
    cols = {p2c["avGFP"].get(str(p)) for p in AVGFP_FORBIDDEN}
    cols.discard(None)
    return {int(p) for p, c in p2c[backbone].items() if c in cols}


def build_oracle(wt_sequences, alignment):
    f = OUT_DIR / "design_oracle.joblib"
    Xd, Xe, Xs, y, meta, sparse_names = load_training()
    if f.exists():
        return joblib.load(f), meta, sparse_names
    dn = np.hstack([Xd, Xe])
    scaler = StandardScaler().fit(dn)
    Xall = sp.hstack([sp.csr_matrix(scaler.transform(dn).astype(np.float32)),
                      Xs]).tocsr()
    print(f"fitting design oracle on all {len(y)} rows, {Xall.shape[1]} features")
    models = fit_ensemble(Xall, y)
    oracle = Oracle(models, scaler, sparse_lookup(sparse_names), Xs.shape[1])
    joblib.dump(oracle, f, compress=3)
    return oracle, meta, sparse_names


def measured_singles(meta, y, backbone):
    """Measured brightness of every single mutant assayed for a backbone."""
    out = {}
    for i, (b, name) in enumerate(zip(meta["Backbone"], meta["Variant name"])):
        if b != backbone:
            continue
        muts = parse_name(name)
        if len(muts) == 1:
            _w, p, m = muts[0]
            out[(p, m)] = float(y[i])
    return out


def phase_design(backbone, wt_sequences, alignment, oracle, meta, y):
    seq = wt_sequences[backbone]
    forbidden = forbidden_positions(backbone, alignment)

    # candidate substitutions = those individually characterised for this
    # backbone, i.e. present anywhere in its training variants
    pool = set()
    for b, name in zip(meta["Backbone"], meta["Variant name"]):
        if b == backbone:
            pool.update((p, m) for _w, p, m in parse_name(name))
    pool = sorted((p, m) for p, m in pool
                  if p not in forbidden and 1 <= p <= len(seq)
                  and seq[p - 1] != m)
    print(f"{backbone}: {len(pool)} characterised substitutions "
          f"({len(forbidden)} chromophore positions excluded)")

    singles = [(backbone, [(seq[p - 1], p, m)]) for p, m in pool]
    X, _raw, names, _ = oracle.featurise(singles, wt_sequences, alignment)
    mean, sd = oracle.predict(X)
    meas = measured_singles(meta, y, backbone)

    srows = []
    for (p, m), mu, s in zip(pool, mean, sd):
        srows.append({"backbone": backbone, "variant": f"{seq[p-1]}{p}{m}",
                      "position": p, "wt_aa": seq[p - 1], "mut_aa": m,
                      "pred_brightness": float(mu), "ens_sd": float(s),
                      "measured_single": meas.get((p, m), np.nan)})
    sdf = pd.DataFrame(srows).sort_values("pred_brightness", ascending=False)
    sdf["rank"] = np.arange(1, len(sdf) + 1)

    # Extension pool: substitutions individually MEASURED at >= MIN_COMPONENT,
    # ranked by that measurement rather than by the oracle. The oracle's
    # single-mutant predictions are heavily compressed toward the mean (K158G
    # measures 2.48 and is predicted 0.917), so where a direct measurement
    # exists it is the better ground truth for a component; the oracle's job
    # here is ranking the combinations, which is what it was validated for.
    elig = sdf.dropna(subset=["measured_single"])
    elig = elig[elig["measured_single"] >= MIN_COMPONENT]
    top = elig.nlargest(EXTEND_POOL, "measured_single")
    ext = [(r.wt_aa, int(r.position), r.mut_aa) for r in top.itertuples()]
    print(f"  extension pool: {len(ext)} substitutions measured "
          f">= {MIN_COMPONENT}x WT alone "
          f"(of {len(elig)} eligible, {len(sdf)} characterised)")
    beam = [[m] for m in ext[:BEAM_WIDTH]]
    designs, history = [], []

    for depth in range(2, MAX_DEPTH + 1):
        cands, seen = [], set()
        for combo in beam:
            used = {p for _w, p, _m in combo}
            for m in ext:
                if m[1] in used:
                    continue
                new = sorted(combo + [m], key=lambda t: t[1])
                key = variant_name(new)
                if key not in seen:
                    seen.add(key)
                    cands.append(new)
        if not cands:
            break
        X, raw, names, missing = oracle.featurise(
            [(backbone, c) for c in cands], wt_sequences, alignment)
        assert missing == 0, "candidate used an uncharacterised substitution"
        mean, sd = oracle.predict(X)
        order = np.argsort(-mean)[:BEAM_WIDTH]
        beam = [cands[i] for i in order]
        cm = names.index("cons_min")
        for i in order:
            comps = cands[i]
            mvals = [meas.get((p, m), np.nan) for _w, p, m in comps]
            designs.append({
                "backbone": backbone, "variant": variant_name(comps),
                "n_mutations": len(comps),
                "pred_brightness": float(mean[i]), "ens_sd": float(sd[i]),
                "pred_lcb": float(mean[i] - sd[i]),
                "n_components_measured_alone": int(np.sum(~np.isnan(mvals))),
                "mean_measured_component": float(np.nanmean(mvals))
                if np.any(~np.isnan(mvals)) else np.nan,
                "min_measured_component": float(np.nanmin(mvals))
                if np.any(~np.isnan(mvals)) else np.nan,
                "cons_min": float(raw[i, cm]),
                "sequence": apply_mutations(seq, comps),
            })
        history.append({"n_mutations": depth, "n_candidates": len(cands),
                        "best_pred": float(mean[order[0]]),
                        "median_top_beam": float(np.median(mean[order]))})
        print(f"  depth {depth}: {len(cands):6d} candidates, "
              f"best {mean[order[0]]:.3f}", flush=True)

    return sdf, pd.DataFrame(designs), pd.DataFrame(history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="avGFP")
    ap.add_argument("--phase", default="all")
    args = ap.parse_args()
    wt_sequences, alignment = load_refs()
    ph = args.phase

    if ph in ("all", "validate"):
        if not phase_validate(wt_sequences, alignment):
            print("FEATURE MISMATCH — stopping.")
            return

    if ph in ("all", "benchmark"):
        f = OUT_DIR / "design_regime_benchmark.csv"
        if not f.exists():
            df, boot = phase_benchmark()
            df.to_csv(f, index=False)
            boot.to_csv(OUT_DIR / "design_regime_bootstrap.csv", index=False)
        df = pd.read_csv(f)
        piv = df[df["k"].isin([10, 20, 50])].pivot_table(
            index=["threshold", "k"], columns="model", values="precision")
        print("\ndesign-regime precision@k:\n" + piv.round(3).to_string())

    if ph in ("all", "design"):
        oracle, meta, _sn = build_oracle(wt_sequences, alignment)
        y = np.load(FEAT_DIR / "y.npy")
        sf = OUT_DIR / "design_singles.csv"
        df_f = OUT_DIR / "design_shortlist.csv"
        if not df_f.exists():
            sdf, designs, hist = phase_design(args.backbone, wt_sequences,
                                              alignment, oracle, meta, y)
            sdf.to_csv(sf, index=False)
            designs.to_csv(df_f, index=False)
            hist.to_csv(OUT_DIR / "design_beam_history.csv", index=False)
        sdf = pd.read_csv(sf)
        designs = pd.read_csv(df_f).sort_values("pred_brightness",
                                                ascending=False)
        print(f"\ntop 10 characterised single substitutions:")
        print(sdf.head(10)[["variant", "pred_brightness", "ens_sd",
                            "measured_single"]].to_string(index=False))
        print(f"\ntop 15 recombination designs:")
        print(designs.head(15)[
            ["variant", "n_mutations", "pred_brightness", "ens_sd",
             "min_measured_component"]].to_string(index=False))


if __name__ == "__main__":
    main()
