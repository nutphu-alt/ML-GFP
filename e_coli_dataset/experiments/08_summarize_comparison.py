"""
GFP variants dataset: step 4b — leaderboard from the model comparison.

Selects each model's best hyperparameter configuration by VALIDATION
Spearman (test is never used for selection), then reports that config's
test metrics. Produces a leaderboard CSV and a comparison figure.

Run with: python summarize_comparison.py
Input : output/model_comparison.csv
Output: output/model_leaderboard.csv, output/model_comparison.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"

MODEL_ORDER = ["ridge", "lasso", "elastic_net", "linear_svr", "svr_rbf",
               "random_forest", "xgboost", "mlp"]


def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Best config per (split, tier, model), chosen on validation Spearman."""
    best_idx = df.groupby(["split", "tier", "model"])["val_spearman"].idxmax()
    best = df.loc[best_idx].copy()
    best["model"] = pd.Categorical(best["model"], MODEL_ORDER, ordered=True)
    return best.sort_values(["split", "tier", "model"])


def main() -> None:
    df = pd.read_csv(OUT_DIR / "model_comparison.csv")
    print(f"total configs evaluated: {len(df)}")

    # 06 never trains the MLP at full scale -- 07 does, and appends its row to
    # this same CSV. This script used to be numbered 07 and so ran first, which
    # silently dropped the best model on the random split from the leaderboard
    # and the figure; the two were swapped so numeric order is now correct.
    # This check is the backstop for anyone running them out of order by hand.
    have = {(r.split, r.model) for r in
            df[df["tier"] == "full"].itertuples()}
    missing = [s for s in ("random", "position") if (s, "mlp") not in have]
    if missing:
        raise SystemExit(
            "model_comparison.csv has no full-tier MLP row for: "
            + ", ".join(missing) + ".\n"
            "06 does not train the MLP at full scale; 07 does. Run it first:\n"
            + "".join(f"  python 07_train_mlp_full.py --split {s}\n" for s in missing)
            + "then re-run this script. (Writing the summary now would understate\n"
              "the best model on the random split by about 0.06 rho.)")

    best = build_leaderboard(df)
    cols = ["split", "tier", "model", "config", "n_train",
            "val_spearman", "test_spearman", "test_pearson", "test_rmse",
            "test_r2", "fit_seconds"]
    best[cols].to_csv(OUT_DIR / "model_leaderboard.csv", index=False)

    for split in ["random", "position"]:
        for tier in ["full", "sub12k"]:
            sub = best[(best["split"] == split) & (best["tier"] == tier)]
            if len(sub):
                print(f"\n=== {split} split / tier={tier} ===")
                print(sub[["model", "config", "n_train", "test_spearman",
                           "test_r2", "fit_seconds"]].to_string(index=False))

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, split in zip(axes, ["random", "position"]):
        sub = best[best["split"] == split]
        models = [m for m in MODEL_ORDER if m in set(sub["model"])]
        x = np.arange(len(models))
        for offset, tier, colour in [(-0.2, "full", "tab:blue"),
                                     (0.2, "sub12k", "tab:orange")]:
            vals = []
            for m in models:
                row = sub[(sub["tier"] == tier) & (sub["model"] == m)]
                vals.append(row["test_spearman"].iloc[0] if len(row) else np.nan)
            ax.bar(x + offset, vals, 0.4, label=f"tier={tier}", color=colour)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        ax.set_title(f"{split} split — test Spearman ρ")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Spearman ρ")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model_comparison.png", dpi=130)
    print("\nwrote model_leaderboard.csv and model_comparison.png")


if __name__ == "__main__":
    main()
