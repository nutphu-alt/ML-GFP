"""
GFP variants dataset: step 3c — analyse baseline results.

The headline number on the position-holdout split is misleading on its
own, because most test rows are multi-mutants that carry only ONE
mutation at a held-out position while their other mutations sit at
positions the model trained on. This script separates those cases so the
true extrapolation performance is visible.

Run with: python analyze_results.py
Input : output/metrics_*.csv, output/preds_*.npz, output/features/meta.csv
Output: output/baseline_results.png, output/extrapolation_breakdown.csv
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"

MUT_RE = re.compile(r"[A-Z](\d+)[A-Z]")
MODELS = ["mean_by_backbone", "ridge_sparse", "ridge_dense", "ridge_combined", "hgb_dense"]


def load_meta() -> pd.DataFrame:
    meta = pd.read_csv(OUT_DIR / "features" / "meta.csv")
    meta["positions"] = meta["Variant name"].apply(
        lambda n: [int(p) for p in MUT_RE.findall(str(n))]
    )
    meta["n_mut"] = meta["positions"].apply(len)
    return meta


def novelty_breakdown(meta: pd.DataFrame) -> pd.DataFrame:
    """For the position split, how much of each test row is genuinely unseen."""
    split = meta["split_position_holdout"].to_numpy()

    # every position appearing in a training row is, by construction, a
    # position the model was allowed to learn
    seen = {
        backbone: set().union(*group["positions"]) if len(group) else set()
        for backbone, group in meta[split == "train"].groupby("Backbone")
    }

    test = meta[split == "test"].copy()
    test["n_novel"] = [
        sum(p not in seen[b] for p in ps)
        for b, ps in zip(test["Backbone"], test["positions"])
    ]
    test["frac_novel"] = test["n_novel"] / test["n_mut"].clip(lower=1)
    return test


def main() -> None:
    meta = load_meta()
    test = novelty_breakdown(meta)

    preds = {s: np.load(OUT_DIR / f"preds_{s}.npz") for s in ["random", "position"]}
    metrics = {s: pd.read_csv(OUT_DIR / f"metrics_{s}.csv", index_col="model")
               for s in ["random", "position"]}

    y = preds["position"]["y_true"]

    # ---- extrapolation breakdown -----------------------------------------
    subsets = {
        "all test rows": np.ones(len(test), bool),
        "single mutant (pure extrapolation)": (test["n_mut"] == 1).to_numpy(),
        "every mutation novel": (test["n_novel"] == test["n_mut"]).to_numpy(),
        "1 novel + others seen": ((test["n_novel"] == 1) & (test["n_mut"] > 1)).to_numpy(),
    }
    rows = []
    for label, mask in subsets.items():
        if mask.sum() > 10:
            row = {"subset": label, "n": int(mask.sum())}
            for model in MODELS:
                row[model] = spearmanr(y[mask], preds["position"][model][mask]).statistic
            rows.append(row)
    breakdown = pd.DataFrame(rows).set_index("subset")
    breakdown.to_csv(OUT_DIR / "extrapolation_breakdown.csv")
    print(breakdown.round(4).to_string())

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # panel 1: Spearman by model and split
    x = np.arange(len(MODELS))
    axes[0].bar(x - 0.2, [metrics["random"].loc[m, "spearman"] for m in MODELS],
                0.4, label="random split")
    axes[0].bar(x + 0.2, [metrics["position"].loc[m, "spearman"] for m in MODELS],
                0.4, label="position-holdout split")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.replace("_", "\n") for m in MODELS], fontsize=8)
    axes[0].set_ylabel("Spearman ρ")
    axes[0].set_title("Test-set ranking performance")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1)

    # panel 2: predicted vs actual for the best model
    best = preds["position"]["ridge_combined"]
    axes[1].hexbin(y, best, gridsize=60, bins="log", cmap="viridis", extent=(0, 1.6, -0.3, 1.6))
    axes[1].plot([0, 1.6], [0, 1.6], "r--", lw=1)
    axes[1].set_xlabel("actual brightness (fold-WT)")
    axes[1].set_ylabel("predicted")
    axes[1].set_title("ridge_combined, position-holdout test")

    # panel 3: performance decays as more of the variant is unseen
    labels, values = [], []
    for k in range(1, 6):
        mask = (test["n_novel"] == k).to_numpy()
        if mask.sum() > 50:
            labels.append(f"{k}")
            values.append(spearmanr(y[mask], best[mask]).statistic)
    axes[2].bar(labels, values, color="tab:orange")
    axes[2].set_xlabel("# mutations at never-seen positions")
    axes[2].set_ylabel("Spearman ρ")
    axes[2].set_title("More novelty → worse prediction")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "baseline_results.png", dpi=130)
    print(f"\nwrote baseline_results.png and extrapolation_breakdown.csv")


if __name__ == "__main__":
    main()
