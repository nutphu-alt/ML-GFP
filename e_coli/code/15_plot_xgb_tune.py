"""Figure for step 8: XGBoost retuning attempt for extrapolation."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(r"/sessions/bold-admiring-volta/mnt/ML for GFP/e_coli/output")

results = pd.read_csv(OUT_DIR / "xgb_tune_results.csv").set_index("config")
boot = pd.read_csv(OUT_DIR / "xgb_tune_bootstrap.csv").set_index("config")

order = ["default", "cand_8", "cand_12", "cand_13"]
short_labels = {
    "default": "default\n(depth 8)",
    "cand_8": "cand_8\n(depth 3)",
    "cand_12": "cand_12\n(depth 4)",
    "cand_13": "cand_13\n(depth 4)",
}
labels = {
    "default": "default (depth 8, lr 0.1, 400 trees)",
    "cand_8": "cand_8 (depth 3, lr 0.1, 300 trees)",
    "cand_12": "cand_12 (depth 4, lr 0.03, 300 trees)",
    "cand_13": "cand_13 (depth 4, lr 0.03, 300 trees, min_child_wt 5)",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

ax = axes[0]
vals = [results.loc[c, "spearman_single"] for c in order]
colors = ["#2C5F2D"] + ["#6FDC3C"] * 3
bars = ax.bar(range(len(order)), vals, color=colors, width=0.6)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([short_labels[c] for c in order], fontsize=9)
ax.set_ylim(0.40, 0.46)
ax.set_ylabel("Pooled out-of-fold Spearman ρ\n(single mutants, n=4,596)")
ax.set_title("Phase 2: all 5 folds pooled")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}",
            ha="center", fontsize=9)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
cands = ["cand_8", "cand_12", "cand_13"]
deltas = [boot.loc[c, "delta_mean"] for c in cands]
los = [boot.loc[c, "delta_mean"] - boot.loc[c, "ci_lo"] for c in cands]
his = [boot.loc[c, "ci_hi"] - boot.loc[c, "delta_mean"] for c in cands]
y = np.arange(len(cands))
ax.errorbar(deltas, y, xerr=[los, his], fmt="o", color="#2C5F2D",
           capsize=4, markersize=8)
ax.axvline(0, color="black", lw=1)
ax.set_yticks(y)
ax.set_yticklabels([labels[c] for c in cands], fontsize=8)
ax.set_xlabel("Δ Spearman ρ vs default (95% CI)")
ax.set_title("Every candidate ties the default\n(all CIs cross zero)")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(-0.025, 0.02)

plt.tight_layout()
plt.savefig(OUT_DIR / "xgb_tune.png", dpi=130)
print("wrote", OUT_DIR / "xgb_tune.png")
