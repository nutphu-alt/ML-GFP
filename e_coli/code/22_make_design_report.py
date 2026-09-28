"""
GFP variants dataset: step 10b — assemble the design panel and figure.

Builds the deliverable an experimentalist actually receives:

  output/design_panel.csv   the ordered test panel, stratified by mutation
                            count, every component individually measured, with
                            the empirical base rate for its depth attached
  output/design.png         four panels covering the whole argument of step 10

Run with: python make_design_report.py [--backbone avGFP] [--per-depth 5]
"""

import argparse
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
FEAT_DIR = OUT_DIR / "features"
CV_DIR = OUT_DIR / "extrap_cv"

MUT_RE = re.compile(r"([A-Z])(\d+)([A-Z])")
MIN_COMPONENT = 0.95

# validated categorical palette (dataviz six-checks: all pass, light surface).
# The orange carries a contrast WARN, so every mark using it is directly
# labelled rather than relying on the swatch alone.
C = {"blue": "#3366CC", "orange": "#EE7733", "green": "#117733",
     "purple": "#AA3377", "grey": "#8C8C8C", "ink": "#222222"}

LITERATURE = {"V163A": "cycle-3", "Y39N": "superfolder", "I171V": "superfolder",
              "Y145F": "superfolder", "N105T": "superfolder", "F99S": "cycle-3",
              "M153T": "cycle-3", "A206V": "monomerising"}


def parse(name):
    return [(w, int(p), m) for w, p, m in MUT_RE.findall(str(name))]


def load_core():
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    muts = [parse(n) for n in meta["Variant name"]]
    nmut = np.array([len(m) for m in muts])
    return y, meta, muts, nmut


def stratum_spearman():
    """Panel (a): what the extrapolation-regime model actually ranks."""
    pred, yt = [], []
    for k in range(5):
        d = np.load(CV_DIR / f"evo_fold{k}.npz", allow_pickle=True)
        pred.append(d["pred"]); yt.append(d["y_true"])
    pred, yt = np.concatenate(pred), np.concatenate(yt)
    out = [("all\nvariants", spearmanr(pred, yt).statistic, len(yt))]
    for lo, lab in [(0.5, "functional\ny>=0.5"), (0.8, "y>=0.8"), (1.0, "y>=1.0")]:
        m = yt >= lo
        out.append((lab, spearmanr(pred[m], yt[m]).statistic, int(m.sum())))
    return out


def component_curves(y, meta, muts, nmut, backbone):
    """Panel (c): brightness vs depth, unfiltered vs component-constrained."""
    av = (meta["Backbone"] == backbone).to_numpy()
    single = {}
    for i in np.where(av & (nmut == 1))[0]:
        _w, p, m = muts[i][0]
        single[(p, m)] = y[i]

    depths = list(range(2, 7))
    curves = {}
    for thr, lab in [(None, "unfiltered"), (0.9, "all components >=0.9x"),
                     (1.0, "all components >=1.0x")]:
        means, ns = [], []
        for k in depths:
            idx = np.where(av & (nmut == k))[0]
            if thr is not None:
                idx = [i for i in idx
                       if all((p, m) in single for _w, p, m in muts[i])
                       and all(single[(p, m)] >= thr for _w, p, m in muts[i])]
            means.append(float(y[list(idx)].mean()) if len(idx) >= 15 else np.nan)
            ns.append(len(idx))
        curves[lab] = (means, ns)
    return depths, curves, single


def base_rates(y, meta, muts, nmut, backbone, single, thr=0.9):
    """Empirical P(y>=1.2) by depth, conditioned on good components."""
    av = (meta["Backbone"] == backbone).to_numpy()
    out = {}
    for k in range(2, 7):
        idx = [i for i in np.where(av & (nmut == k))[0]
               if all((p, m) in single for _w, p, m in muts[i])
               and all(single[(p, m)] >= thr for _w, p, m in muts[i])]
        out[k] = (float((y[idx] >= 1.2).mean()) if len(idx) >= 15 else np.nan,
                  len(idx))
    return out


def build_panel(designs, singles, rates, per_depth):
    rows = []
    for k, block in designs.groupby("n_mutations"):
        for r in block.nlargest(per_depth, "pred_brightness").itertuples():
            rate, n = rates.get(int(k), (np.nan, 0))
            rows.append({
                "tier": f"{int(k)}-mutation design", "variant": r.variant,
                "n_mutations": int(k),
                "pred_brightness": round(r.pred_brightness, 3),
                "ens_sd": round(r.ens_sd, 3),
                "min_measured_component": round(r.min_measured_component, 3),
                "mean_measured_component": round(r.mean_measured_component, 3),
                "empirical_P_ge_1.2x_for_this_depth": round(rate, 3),
                "n_supporting_measured_variants": n,
                "sequence": r.sequence,
            })
    # single-mutant controls: the components the model rates highest that are
    # also individually measured bright. These anchor the plate.
    ctrl = singles.dropna(subset=["measured_single"])
    ctrl = ctrl[ctrl["measured_single"] >= 1.2].nlargest(3, "pred_brightness")
    for r in ctrl.itertuples():
        rows.append({
            "tier": "single-mutant control", "variant": r.variant,
            "n_mutations": 1,
            "pred_brightness": round(r.pred_brightness, 3),
            "ens_sd": round(r.ens_sd, 3),
            "min_measured_component": round(r.measured_single, 3),
            "mean_measured_component": round(r.measured_single, 3),
            "empirical_P_ge_1.2x_for_this_depth": np.nan,
            "n_supporting_measured_variants": 1,
            "sequence": "",
        })
    return pd.DataFrame(rows)


def make_figure(strata, bench, depths, curves, panel, rates, backbone):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#E3E3E0", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#C9C9C4")

    # (a) what the extrapolation model ranks
    ax = axes[0, 0]
    labs = [s[0] for s in strata]; vals = [s[1] for s in strata]
    cols = [C["blue"] if v > 0.3 else C["purple"] for v in vals]
    b = ax.bar(labs, vals, color=cols, width=0.6)
    for rect, (_l, v, n) in zip(b, strata):
        ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2,
                                 v + (0.02 if v >= 0 else -0.05)),
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9, color=C["ink"])
        ax.annotate(f"n={n:,}", (rect.get_x() + rect.get_width() / 2, -0.272),
                    ha="center", fontsize=7.5, color="#6B6B66")
    ax.axhline(0, color="#9A9A94", lw=1)
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title("(a) At never-assayed positions the model ranks\n"
                 "functionality, not brightness", fontsize=11, loc="left")
    ax.set_ylim(-0.30, 0.56)

    # (b) design-regime precision@20
    ax = axes[0, 1]
    sub = bench[bench["k"] == 20]
    models = sorted(sub["model"].unique(), key=len)
    thrs = [1.0, 1.2, 1.5]
    w, x = 0.36, np.arange(len(thrs))
    for i, mdl in enumerate(models):
        v = [float(sub[(sub["model"] == mdl) & (sub["threshold"] == t)]["precision"].iloc[0])
             for t in thrs]
        colr = C["grey"] if "sparse" not in mdl else C["green"]
        lab = "dense+evo (77)" if "sparse" not in mdl else "dense+evo+sparse (6,636)"
        rects = ax.bar(x + (i - 0.5) * w, v, w * 0.92, color=colr, label=lab)
        for r, vv in zip(rects, v):
            ax.annotate(f"{vv:.2f}", (r.get_x() + r.get_width() / 2, vv + 0.02),
                        ha="center", fontsize=8.5, color=C["ink"])
    for j, t in enumerate(thrs):
        base = float(sub[sub["threshold"] == t]["baseline"].iloc[0])
        ax.plot([j - 0.55, j + 0.55], [base, base], color=C["ink"], lw=1.4,
                ls=(0, (3, 2)))
    ax.annotate("dashed = base rate", (-0.55, 0.20), fontsize=8, color=C["ink"],
                ha="left")
    ax.set_xticks(x, [f"y $\\geq$ {t}x WT" for t in thrs])
    ax.set_ylabel("precision @ top 20")
    ax.set_ylim(0, 1.12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("(b) In the recombination regime it ranks brightness well —\n"
                 "and the retired sparse block is what finds the brightest",
                 fontsize=11, loc="left")

    # (c) the component constraint
    ax = axes[1, 0]
    style = {"unfiltered": (C["purple"], "o"),
             "all components >=0.9x": (C["orange"], "s"),
             "all components >=1.0x": (C["blue"], "^")}
    for lab, (means, ns) in curves.items():
        colr, mk = style[lab]
        ax.plot(depths, means, color=colr, lw=2, marker=mk, ms=7,
                mec="#fcfcfb", mew=1.5, label=lab)
        last = max([i for i, v in enumerate(means) if not np.isnan(v)])
        ax.annotate(lab.replace("all components ", ""),
                    (depths[last] + 0.08, means[last]), color=colr, fontsize=9,
                    va="center")
    ax.set_xlabel("mutations per variant")
    ax.set_ylabel("mean brightness (fold WT)")
    ax.set_xticks(depths)
    ax.set_xlim(1.8, 7.2)
    ax.set_title("(c) Brightness collapses with stacking only when components\n"
                 "are unscreened — the constraint that makes design work",
                 fontsize=11, loc="left")

    # (d) the panel
    ax = axes[1, 1]
    d = panel[panel["tier"] != "single-mutant control"]
    ax.errorbar(d["n_mutations"] + np.random.default_rng(0).normal(0, 0.05, len(d)),
                d["pred_brightness"], yerr=d["ens_sd"], fmt="o", ms=7,
                color=C["blue"], ecolor="#B9C7E8", elinewidth=1.6, capsize=0,
                mec="#fcfcfb", mew=1.2, label="design (prediction $\\pm$ seed sd)")
    c = panel[panel["tier"] == "single-mutant control"]
    ax.scatter(c["n_mutations"], c["min_measured_component"], marker="D", s=52,
               color=C["green"], ec="#fcfcfb", lw=1.2, zorder=5,
               label="single-mutant control (measured)")
    for r in c.itertuples():
        ax.annotate(r.variant, (r.n_mutations + 0.12, r.min_measured_component),
                    fontsize=8.5, color=C["green"], va="center")
    ax2 = ax.twiny()      # a second x-axis of labels, not a second scale
    ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(sorted(d["n_mutations"].unique()))
    ax2.set_xticklabels([f"{rates[k][0]:.0%}" if not np.isnan(rates[k][0]) else "—"
                         for k in sorted(d["n_mutations"].unique())], fontsize=8.5,
                        color="#6B6B66")
    ax2.set_xlabel("empirical P(y $\\geq$ 1.2x) at this depth, screened components",
                   fontsize=8.5, color="#6B6B66")
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax.set_xlabel("mutations per design")
    ax.set_ylabel("predicted brightness (fold WT)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"(d) The proposed {backbone} panel", fontsize=11, loc="left")

    fig.suptitle("Step 10 — designing bright GFP variants by recombination",
                 fontsize=13.5, x=0.055, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "design.png", dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {OUT_DIR / 'design.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="avGFP")
    ap.add_argument("--per-depth", type=int, default=5)
    args = ap.parse_args()

    y, meta, muts, nmut = load_core()
    designs = pd.read_csv(OUT_DIR / "design_shortlist.csv")
    singles = pd.read_csv(OUT_DIR / "design_singles.csv")
    bench = pd.read_csv(OUT_DIR / "design_regime_benchmark.csv")

    strata = stratum_spearman()
    depths, curves, single = component_curves(y, meta, muts, nmut, args.backbone)
    rates = base_rates(y, meta, muts, nmut, args.backbone, single)

    panel = build_panel(designs, singles, rates, args.per_depth)
    panel.to_csv(OUT_DIR / "design_panel.csv", index=False)
    print(f"wrote {OUT_DIR / 'design_panel.csv'} ({len(panel)} variants)\n")

    show = panel[["tier", "variant", "pred_brightness", "ens_sd",
                  "min_measured_component", "empirical_P_ge_1.2x_for_this_depth"]]
    print(show.to_string(index=False))

    lit = [v for v in singles.head(60)["variant"] if v in LITERATURE]
    print(f"\nliterature mutations inside the model's top 60 single "
          f"substitutions: {', '.join(lit)}")

    make_figure(strata, bench, depths, curves, panel, rates, args.backbone)


if __name__ == "__main__":
    main()
