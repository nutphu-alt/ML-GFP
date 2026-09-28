"""
GFP variants dataset: step 11e — figure for the ESM-2 result.

Run with: python make_esm_report.py
Output: output/esm.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"

# validated categorical palette, reused from make_design_report.py
C = {"blue": "#3366CC", "orange": "#EE7733", "green": "#117733",
     "purple": "#AA3377", "grey": "#8C8C8C", "ink": "#222222"}


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#E3E3E0", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C9C9C4")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        style(ax)

    # (a) zero-shot score vs embedding, single-mutant subset
    ax = axes[0, 0]
    arms = [("dense\n(66)", 0.4377, C["grey"]),
            ("+ evo\n(77)", 0.4785, C["blue"]),
            ("+ ESM-2\nscore (81)", 0.4870, C["orange"]),
            ("+ ESM-2\nembed (109)", 0.5185, C["green"])]
    xs = np.arange(len(arms))
    vals = [a[1] for a in arms]
    cols = [a[2] for a in arms]
    b = ax.bar(xs, vals, color=cols, width=0.62)
    for rect, v in zip(b, vals):
        ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v + 0.008),
                    ha="center", fontsize=9.5, color=C["ink"])
    ax.set_xticks(xs, [a[0] for a in arms], fontsize=9.5)
    ax.set_ylabel("Spearman $\\rho$, single mutants\n(never-assayed positions)")
    ax.set_ylim(0, 0.62)
    ax.set_title("(a) Zero-shot scores add little; embeddings\n"
                 "add what the evo features add, again",
                 fontsize=11, loc="left")
    ax.annotate("embed vs +evo: +0.040, 95% CI [+0.025, +0.055]",
               (3, 0.565), ha="center", fontsize=8.5, color=C["ink"])

    # (b) LOBO by backbone — the ppluGFP2 story
    ax = axes[0, 1]
    lobo = pd.read_csv(OUT_DIR / "lobo_esm.csv").set_index("features")
    backbones = ["amacGFP", "avGFP", "cgreGFP", "ppluGFP2"]
    w, x = 0.36, np.arange(len(backbones))
    for i, (label, color) in enumerate([("dense+evo (77)", C["blue"]),
                                        ("dense+evo+emb (109)", C["green"])]):
        v = [lobo.loc[label, b] for b in backbones]
        rects = ax.bar(x + (i - 0.5) * w, v, w * 0.92, color=color, label=label)
        for r, vv in zip(rects, v):
            dy = 0.028 if i == 0 else 0.008
            ax.annotate(f"{vv:.2f}", (r.get_x() + r.get_width() / 2, vv + dy),
                        ha="center", fontsize=8.5, color=C["ink"])
    ax.set_xticks(x, backbones, fontsize=9.5)
    ax.set_ylabel("Spearman $\\rho$, held out as the test backbone")
    ax.set_ylim(0, 0.80)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.02))
    ax.set_title("(b) ppluGFP2 — the outlier the evo features hurt —\n"
                 "is the one embeddings fix (0.31 $\\to$ 0.39)",
                 fontsize=11, loc="left")
    ax.annotate("phylogenetic\noutlier", (3, 0.16), ha="center", fontsize=8,
               color=C["purple"])

    # (c) dims sweep — flat
    ax = axes[1, 0]
    sweep = pd.read_csv(OUT_DIR / "esm_dims_sweep.csv")
    ax.plot(sweep["dims"], sweep["rho_single"], color=C["blue"], lw=2,
           marker="o", ms=8, mec="#fcfcfb", mew=1.5)
    for _, r in sweep.iterrows():
        ax.annotate(f"{r['rho_single']:.3f}", (r["dims"], r["rho_single"] + 0.004),
                    ha="center", fontsize=8.5, color=C["ink"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(sweep["dims"], sweep["dims"])
    ax.set_xlabel("PCA components kept from the 1280-dim embedding")
    ax.set_ylabel("Spearman $\\rho$, single mutants")
    ax.set_ylim(0.5, 0.53)
    ax.set_title("(c) Width doesn't matter — every setting's 95% CI\n"
                 "vs 16 dims spans zero. 16 is kept.",
                 fontsize=11, loc="left")

    # (d) how much of the design result rests on near neighbours
    ax = axes[1, 1]
    st = pd.read_csv(OUT_DIR / "design_stress_test.csv")
    sub = st[st["k"] == 20]
    groups = ["has_neighbour", "no_neighbour"]
    short = {"has_neighbour": "near neighbour\nin library",
             "no_neighbour": "no near\nneighbour"}
    thrs = [1.0, 1.2, 1.5]
    w2, x2 = 0.34, np.arange(len(thrs))
    colors = [C["blue"], C["orange"]]
    for i, g in enumerate(groups):
        v = [float(sub[(sub["subset"] == g) & (sub["threshold"] == t)]["precision"].iloc[0])
             for t in thrs]
        rects = ax.bar(x2 + (i - 0.5) * w2, v, w2 * 0.92, color=colors[i],
                       label=short[g])
        for r, vv in zip(rects, v):
            ax.annotate(f"{vv:.2f}", (r.get_x() + r.get_width() / 2, vv + 0.02),
                        ha="center", fontsize=8.5, color=C["ink"])
    for j, t_ in enumerate(thrs):
        base = float(sub[sub["threshold"] == t_]["baseline"].iloc[0])
        ax.plot([j - 0.5, j + 0.5], [base, base], color=C["ink"], lw=1.3,
                ls=(0, (3, 2)))
    ax.set_xticks(x2, [f"y $\\geq$ {t_}x WT" for t_ in thrs])
    ax.set_ylabel("precision @ top 20")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=1)
    ax.set_title("(d) Half the designs have no near neighbour —\n"
                 "those give the honest lower bound",
                 fontsize=11, loc="left")

    fig.suptitle("Steps 11-13 — ESM-2 embeddings, and an honest range for "
                 "the design panel", fontsize=13.5, x=0.055, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "esm.png", dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {OUT_DIR / 'esm.png'}")


if __name__ == "__main__":
    main()
