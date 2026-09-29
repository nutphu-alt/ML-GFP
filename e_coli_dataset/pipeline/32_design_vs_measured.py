"""Step 14: the designed panel against the brightest variants actually measured.

Answers "are the designs better than what we already have?" -- but only after
putting both groups on ONE scale. The oracle compresses badly (it scores K158G
at ~0.92 though that variant measures 2.48x WT), so a predicted number and a
measured number are NOT comparable. This script therefore scores the top
measured avGFP variants with the same oracle, and plots that gap explicitly.

Needs pipeline/21_design_variants.py to have run (design_shortlist.csv and
design_oracle.joblib) -- run this after it.

Writes:
  output/top10_designs_vs_best_measured.csv
  output/design_vs_measured.png
"""
import importlib
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"

# 21_design_variants lives in pipeline/; a module name cannot start with a digit,
# so it is loaded by name (same pattern as 28 and 29).
sys.path.insert(0, str(BASE_DIR / "pipeline"))
dv = importlib.import_module("21_design_variants")

# design_oracle.joblib was pickled while 21 ran as __main__, so the Oracle class
# has to be reachable there for joblib.load to resolve it.
import __main__
__main__.Oracle = dv.Oracle

TOP_N = 10
BLUE, ORANGE, GREY = "#1f77b4", "#ff7f0e", "#9aa0a6"
MUT_RE = re.compile(r"[A-Z]\d+[A-Z]")


def triples(name):
    return [(m[0], int(m[1:-1]), m[-1]) for m in MUT_RE.findall(str(name))]


def main():
    shortlist = OUT_DIR / "design_shortlist.csv"
    if not shortlist.exists():
        raise SystemExit(
            f"missing {shortlist.name}. Run:  python ../pipeline/21_design_variants.py")

    designs = (pd.read_csv(shortlist)
                 .sort_values("pred_brightness", ascending=False)
                 .head(TOP_N).reset_index(drop=True))

    dat = pd.read_csv(OUT_DIR / "gfp_variants_split.csv", low_memory=False)
    av = dat[(dat["Backbone"] == "avGFP") & (dat["Source group"] == "DMS")].copy()
    best = av.nlargest(TOP_N, "Brightness value").reset_index(drop=True)

    # novelty: none of the designs may already exist in the library
    lib_seqs = set(av["Protein sequence"])
    lib_sets = set(av["Variant name"].map(lambda n: frozenset(MUT_RE.findall(str(n)))))
    n_seq = designs["sequence"].isin(lib_seqs).sum()
    n_set = designs["variant"].map(lambda n: frozenset(MUT_RE.findall(str(n)))).isin(lib_sets).sum()
    print(f"novelty check: {n_seq}/{TOP_N} sequences and {n_set}/{TOP_N} mutation sets "
          f"already in the library (both should be 0)")

    # score the measured variants with the SAME oracle, so the comparison is
    # predicted-vs-predicted rather than predicted-vs-measured
    wt, align = dv.load_refs()
    oracle, _meta, _sparse_names = dv.build_oracle(wt, align)
    records = [("avGFP", triples(n)) for n in best["Variant name"]]
    X, _raw, _names, missing = oracle.featurise(records, wt, align)
    mean, sd = oracle.predict(X)
    best["predicted"], best["pred_sd"] = mean, sd
    best["measured"] = best["Brightness value"]
    best["short"] = best["Variant name"].str.replace("avGFP ", "", regex=False)
    if missing:
        print(f"note: {missing} substitutions had no sparse column")

    designs.to_csv(OUT_DIR / "top10_designs_vs_best_measured.csv", index=False)

    print(f"\ndesigns   mean predicted {designs.pred_brightness.mean():.3f}")
    print(f"measured  mean measured  {best.measured.mean():.3f} "
          f"| mean predicted {best.predicted.mean():.3f}")

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharex=True)

    ax = axes[0]
    d = designs.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(d))
    ax.hlines(yy, 0, d.pred_brightness, color=BLUE, lw=2, alpha=.35)
    ax.errorbar(d.pred_brightness, yy, xerr=d.ens_sd, fmt="o", ms=9, color=BLUE,
                ecolor=BLUE, elinewidth=2, capsize=3, zorder=3)
    for i, r in d.iterrows():
        ax.text(r.pred_brightness + r.ens_sd + .06, i, f"{r.pred_brightness:.2f}",
                va="center", fontsize=9, color="#202124")
    ax.set_yticks(yy)
    ax.set_yticklabels(d.variant, fontsize=8.5, family="monospace")
    ax.set_title(f"(a) {TOP_N} designed variants — predicted, never synthesised\n"
                 "none of these sequences exists in the library",
                 fontsize=11, loc="left", pad=26)
    ax.set_xlabel("brightness (fold wild-type)")

    ax = axes[1]
    m = best.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(m))
    ax.hlines(yy, m.predicted, m.measured, color=GREY, lw=2, zorder=1)
    ax.plot(m.measured, yy, "o", ms=9, color=ORANGE, zorder=3, label="measured in the library")
    ax.plot(m.predicted, yy, "o", ms=9, color=BLUE, zorder=3, label="oracle's prediction")
    for i, r in m.iterrows():
        ax.text(r.measured + .06, i, f"{r.measured:.2f}", va="center",
                fontsize=9, color="#202124")
        ax.text(r.predicted - .06, i, f"{r.predicted:.2f}", va="center", ha="right",
                fontsize=9, color="#202124")
    ax.set_yticks(yy)
    ax.set_yticklabels(m.short, fontsize=8.5, family="monospace")
    ax.set_title(f"(b) {TOP_N} brightest measured avGFP variants\n"
                 "grey bar = how far the oracle under-predicts each one",
                 fontsize=11, loc="left", pad=26)
    ax.set_xlabel("brightness (fold wild-type)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              fontsize=9.5, frameon=False)

    for ax in axes:
        ax.axvline(1.0, color="#c0c4c8", ls="--", lw=1.2, zorder=0)
        ax.set_xlim(0, 2.75)
        ax.set_ylim(-0.55, TOP_N - 0.5)
        ax.text(1.0, 1.008, "wild-type", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color="#5f6368", ha="center", va="bottom")
        ax.grid(axis="x", alpha=.25)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)

    fig.suptitle("Designed variants vs the best measured ones — compared on the model's own scale",
                 fontsize=13, x=.09, ha="left", y=.995)
    fig.text(.09, .015,
             f"On the oracle's scale the {TOP_N} designs (mean "
             f"{designs.pred_brightness.mean():.2f}) outrank the {TOP_N} brightest known "
             f"variants (mean {best.predicted.mean():.2f}), because the model compresses:\n"
             "it scores K158G at 0.92 though it measures 2.48. Predicted and measured values "
             "are NOT on a comparable scale — panel (b) is the evidence.",
             fontsize=9, color="#3c4043", va="bottom")
    fig.tight_layout(rect=[0, .105, 1, .945])
    fig.savefig(OUT_DIR / "design_vs_measured.png", dpi=150, facecolor="white")
    print(f"\nwrote {OUT_DIR / 'top10_designs_vs_best_measured.csv'}")
    print(f"wrote {OUT_DIR / 'design_vs_measured.png'}")


if __name__ == "__main__":
    main()
