"""
GFP variants dataset: step 5b — analyse the cross-validated extrapolation benchmark.

Pools the out-of-fold predictions from cross_val_extrapolation.py so every
single mutant in the dataset is scored by a model that never saw its
position. Compares models with a paired bootstrap, which is the right test
here: all models predict the same rows, so resampling rows jointly cancels
the row-difficulty variance that an unpaired comparison would leave in.

Run with: python analyze_extrapolation_cv.py
Output: output/extrapolation_cv_results.csv, output/extrapolation_cv.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
CV_DIR = OUT_DIR / "extrap_cv"

MODELS = ["ridge", "xgboost", "mlp"]
N_FOLDS = 5
N_BOOT = 4000


def load_pooled() -> pd.DataFrame:
    """Stack every fold's out-of-fold predictions into one frame."""
    frames = []
    for k in range(N_FOLDS):
        base = np.load(CV_DIR / f"fold{k}_ridge.npz", allow_pickle=True)
        df = pd.DataFrame({
            "fold": k,
            "row_index": base["row_index"],
            "y": base["y_true"],
            "n_mut": base["n_mut"],
            "backbone": base["backbone"],
        })
        for m in MODELS:
            d = np.load(CV_DIR / f"fold{k}_{m}.npz", allow_pickle=True)
            assert np.array_equal(d["row_index"], base["row_index"]), \
                f"fold {k}: {m} rows do not align with ridge"
            df[m] = d["pred"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def paired_bootstrap(y, pred_a, pred_b, rng) -> tuple:
    """Bootstrap the paired difference in Spearman between two models."""
    n = len(y)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = (spearmanr(y[idx], pred_a[idx]).statistic
                    - spearmanr(y[idx], pred_b[idx]).statistic)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), lo, hi, float((diffs > 0).mean())


def main() -> None:
    df = load_pooled()
    single = df[df["n_mut"] == 1]
    print(f"pooled test rows (all mutations novel): {len(df)}")
    print(f"single mutants at never-seen positions: {len(single)}  "
          f"(previous benchmark: 293)\n")

    rows = []
    for label, sub in [("all-novel variants", df), ("single mutants", single)]:
        r = {"subset": label, "n": len(sub)}
        for m in MODELS:
            r[m] = spearmanr(sub["y"], sub[m]).statistic
        rows.append(r)
    for backbone, sub in single.groupby("backbone"):
        r = {"subset": f"single — {backbone}", "n": len(sub)}
        for m in MODELS:
            r[m] = spearmanr(sub["y"], sub[m]).statistic
        rows.append(r)
    table = pd.DataFrame(rows).set_index("subset")
    print(table.round(4).to_string())
    table.to_csv(OUT_DIR / "extrapolation_cv_results.csv")

    # ---- paired bootstrap on the headline subset --------------------------
    print(f"\npaired bootstrap on {len(single)} single mutants "
          f"({N_BOOT} resamples):")
    rng = np.random.default_rng(0)
    y = single["y"].to_numpy()
    comparisons = [("xgboost", "mlp"), ("xgboost", "ridge"), ("ridge", "mlp")]
    boot_rows = []
    for a, b in comparisons:
        mean, lo, hi, p = paired_bootstrap(
            y, single[a].to_numpy(), single[b].to_numpy(), rng)
        verdict = "significant" if lo > 0 or hi < 0 else "not significant"
        print(f"  {a:8s} − {b:8s}  Δρ={mean:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(Δ>0)={p:.3f}  {verdict}")
        boot_rows.append({"comparison": f"{a} - {b}", "delta_rho": mean,
                          "ci_low": lo, "ci_high": hi, "p_gt_0": p,
                          "verdict": verdict})
    pd.DataFrame(boot_rows).to_csv(
        OUT_DIR / "extrapolation_cv_bootstrap.csv", index=False)

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    subsets = ["all-novel variants", "single mutants"]
    x = np.arange(len(subsets))
    width = 0.26
    colours = {"ridge": "tab:orange", "xgboost": "tab:green", "mlp": "tab:red"}
    for i, m in enumerate(MODELS):
        vals = [table.loc[s, m] for s in subsets]
        axes[0].bar(x + (i - 1) * width, vals, width, label=m, color=colours[m])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s}\n(n={table.loc[s, 'n']:,.0f})" for s in subsets])
    axes[0].set_ylabel("Spearman ρ")
    axes[0].set_title("Out-of-fold performance at never-seen positions")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    labels = [f"{a}\n− {b}" for a, b in comparisons]
    means = [r["delta_rho"] for r in boot_rows]
    errs = [[r["delta_rho"] - r["ci_low"] for r in boot_rows],
            [r["ci_high"] - r["delta_rho"] for r in boot_rows]]
    axes[1].bar(labels, means, 0.5, yerr=errs, capsize=6, color="tab:blue")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_ylabel("Δ Spearman ρ")
    axes[1].set_title(f"Paired bootstrap, {len(single):,} single mutants (95% CI)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "extrapolation_cv.png", dpi=130)
    print("\nwrote extrapolation_cv_results.csv, extrapolation_cv_bootstrap.csv, "
          "extrapolation_cv.png")


if __name__ == "__main__":
    main()
