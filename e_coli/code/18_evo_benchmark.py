"""
GFP variants dataset: step 9c — do evolutionary features break the plateau?

The decisive test for step 9. Same 5-fold position cross-validation as
step 5 (SEED=7, identical fold assignment), same XGBoost configuration
that step 8 confirmed is as good as anything nearby, same pooled
single-mutant metric (n=4,596). The only thing that changes is the feature
block:

    dense       66 transferable descriptors                (the baseline)
    dense+evo   those 66 plus the 11 cross-homolog features

The baseline predictions are not recomputed — step 8 already wrote exactly
this model on exactly these folds to output/extrap_cv/xgbtune_fold{k}_default.npz,
so reusing them makes the comparison exactly paired rather than merely
comparable.

Also reports XGBoost's gain-based feature importance for the evo block, to
distinguish "the features did not help" from "the model ignored them".

Resume-safe: per-fold predictions are written as they finish.

Run with: python evo_benchmark.py
Output: output/extrap_cv/evo_fold{k}.npz, output/evo_benchmark.csv,
        output/evo_bootstrap.csv, output/evo_importance.csv, output/evo.png
"""

import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"
CV_DIR = OUT_DIR / "extrap_cv"

MUT_RE = re.compile(r"[A-Z](\d+)[A-Z]")
N_FOLDS = 5
SEED = 7                  # must match cross_val_extrapolation.py
TIME_BUDGET = 150
N_BOOT = 4000

# the step-8-confirmed configuration, unchanged
XGB_CFG = dict(n_estimators=400, max_depth=8, learning_rate=0.1,
               subsample=0.8, colsample_bytree=0.8)


def assign_position_folds(meta, rng):
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


def run_folds():
    X_dense = np.load(FEAT_DIR / "X_dense.npy")
    X_evo = np.load(FEAT_DIR / "X_evo.npy")
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    evo_names = (FEAT_DIR / "evo_feature_names.txt").read_text().split()
    dense_names = (FEAT_DIR / "dense_feature_names.txt").read_text().split()

    rng = np.random.default_rng(SEED)
    folds, positions = assign_position_folds(meta, rng)

    started = time.time()
    importances = []
    for k in range(N_FOLDS):
        path = CV_DIR / f"evo_fold{k}.npz"
        if path.exists():
            continue
        if time.time() - started > TIME_BUDGET:
            print(f"time budget reached — re-run to continue (next: fold {k})")
            return None
        train, test, n_mut = fold_masks(meta, positions, folds, k)

        # scaler fit on this fold's training rows only, as in step 5
        combined = np.hstack([X_dense, X_evo])
        scaler = StandardScaler().fit(combined[train])
        X = scaler.transform(combined).astype(np.float32)

        t0 = time.time()
        model = xgb.XGBRegressor(tree_method="hist", n_jobs=-1,
                                 random_state=0, **XGB_CFG)
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        np.savez_compressed(
            path, pred=pred, y_true=y[test], n_mut=n_mut[test],
            backbone=meta.loc[test, "Backbone"].to_numpy(),
            importance=model.feature_importances_)
        print(f"fold {k} dense+evo train={train.sum():6d} test={test.sum():5d} "
              f"({time.time() - t0:.0f}s)", flush=True)
        importances.append(model.feature_importances_)

    return dense_names + evo_names, len(evo_names)


def pooled(prefix):
    preds, ys, nmuts, backbones = [], [], [], []
    for k in range(N_FOLDS):
        p = CV_DIR / prefix.format(k=k)
        if not p.exists():
            return None
        d = np.load(p, allow_pickle=True)
        preds.append(d["pred"]); ys.append(d["y_true"])
        nmuts.append(d["n_mut"]); backbones.append(d["backbone"])
    return (np.concatenate(preds), np.concatenate(ys),
            np.concatenate(nmuts), np.concatenate(backbones))


def paired_bootstrap(y_true, pred_a, pred_b, rng):
    """Δρ = ρ(a) − ρ(b) with a percentile CI, resampling rows jointly."""
    n = len(y_true)
    deltas = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        deltas[i] = (spearmanr(y_true[idx], pred_a[idx]).statistic
                     - spearmanr(y_true[idx], pred_b[idx]).statistic)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return deltas.mean(), lo, hi, float((deltas > 0).mean())


def main() -> None:
    out = run_folds()
    if out is None:
        return
    feature_names, n_evo = out

    evo = pooled("evo_fold{k}.npz")
    base = pooled("xgbtune_fold{k}_default.npz")
    if evo is None or base is None:
        print("missing fold predictions — run step 8's eval phase first")
        return

    pred_e, y_e, nmut_e, bb_e = evo
    pred_b, y_b, nmut_b, _ = base
    assert np.allclose(y_e, y_b), "fold row order differs between the two runs"

    rows = []
    subsets = [("all-novel variants", np.ones(len(y_e), dtype=bool)),
               ("single mutants", nmut_e == 1)]
    for backbone in sorted(set(bb_e.tolist())):
        subsets.append((f"single — {backbone}",
                        (nmut_e == 1) & (bb_e == backbone)))

    print("\n=== out-of-fold Spearman at never-seen positions ===")
    print(f"{'subset':28s} {'n':>6s} {'dense':>8s} {'dense+evo':>10s} {'Δ':>8s}")
    for label, mask in subsets:
        r_b = spearmanr(y_b[mask], pred_b[mask]).statistic
        r_e = spearmanr(y_e[mask], pred_e[mask]).statistic
        rows.append({"subset": label, "n": int(mask.sum()),
                     "dense": r_b, "dense_evo": r_e, "delta": r_e - r_b})
        print(f"{label:28s} {mask.sum():6d} {r_b:8.4f} {r_e:10.4f} {r_e - r_b:+8.4f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "evo_benchmark.csv", index=False)

    rng = np.random.default_rng(0)
    boot_rows = []
    print("\n=== paired bootstrap (dense+evo − dense) ===")
    for label, mask in subsets[:2]:
        d, lo, hi, p = paired_bootstrap(y_e[mask], pred_e[mask], pred_b[mask], rng)
        verdict = "significant" if (lo > 0 or hi < 0) else "not significant"
        boot_rows.append({"subset": label, "delta": d, "ci_lo": lo,
                          "ci_hi": hi, "p_gt_0": p, "verdict": verdict})
        print(f"{label:22s} Δρ={d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"P(Δ>0)={p:.3f}  {verdict}")
    pd.DataFrame(boot_rows).to_csv(OUT_DIR / "evo_bootstrap.csv", index=False)

    # did the model actually use the evo block?
    imps = np.mean([np.load(CV_DIR / f"evo_fold{k}.npz")["importance"]
                    for k in range(N_FOLDS)], axis=0)
    imp = pd.DataFrame({"feature": feature_names, "gain": imps})
    imp["block"] = ["evo" if i >= len(feature_names) - n_evo else "dense"
                    for i in range(len(feature_names))]
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(OUT_DIR / "evo_importance.csv", index=False)
    evo_share = imp.loc[imp.block == "evo", "gain"].sum()
    print(f"\nevo block holds {evo_share:.1%} of total feature importance "
          f"({n_evo} of {len(feature_names)} features = "
          f"{n_evo / len(feature_names):.1%} of the width)")
    print("\ntop 12 features overall:")
    print(imp.head(12).to_string(index=False))

    make_figure(rows, boot_rows, imp, n_evo)


def make_figure(rows, boot_rows, imp, n_evo):
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    ax = axes[0]
    sub = df[df.subset.str.startswith("single —") | (df.subset == "single mutants")]
    labels = [s.replace("single — ", "").replace("single mutants", "ALL")
              for s in sub.subset]
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["dense"], 0.4, label="dense (66)", color="#2C5F2D")
    ax.bar(x + 0.2, sub["dense_evo"], 0.4, label="dense + evo (77)", color="#6FDC3C")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Spearman ρ, single mutants")
    ax.set_title("Out-of-fold, never-seen positions")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    b = pd.DataFrame(boot_rows)
    y = np.arange(len(b))
    ax.errorbar(b["delta"], y,
                xerr=[b["delta"] - b["ci_lo"], b["ci_hi"] - b["delta"]],
                fmt="o", color="#2C5F2D", capsize=4, markersize=8)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(b["subset"], fontsize=9)
    ax.set_xlabel("Δ Spearman ρ vs dense-only (95% CI)")
    ax.set_title("Paired bootstrap")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[2]
    top = imp.head(12).iloc[::-1]
    colors = ["#6FDC3C" if b == "evo" else "#B8C9BA" for b in top.block]
    ax.barh(np.arange(len(top)), top.gain, color=colors)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.feature, fontsize=8)
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Top features (green = evolutionary)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "evo.png", dpi=130)
    print(f"\nwrote {OUT_DIR / 'evo.png'}")


if __name__ == "__main__":
    main()
