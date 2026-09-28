# E. coli GFP dataset — pipeline and experiments

Predicting fold-wild-type fluorescence brightness from amino-acid sequence, and
using that predictor to design brighter GFP variants.

Project overview, headline results and setup are in the [repository
README](../README.md). This file covers the layout and how to run it.

## The deliverable

`output/design_panel.csv` — 23 avGFP variants proposed for the bench, stratified
by mutation count, every component substitution individually measured at ≥0.95×
wild-type, and none of them existing library members.

Expected hit rate is a **range, not a point estimate**: precision@20 is 0.65–0.90
for ≥1.2× WT and ~0.35–0.45 for ≥1.5× WT, depending on whether a design has a
near neighbour in the library. Even the pessimistic end is ~21× over the 3.0%
base rate. The design loop independently rediscovers the superfolder GFP
mutation set with no literature input.

## Layout

```
e_coli_dataset/
├── dataset/        15 source .xlsx files (+ README describing the columns)
├── pipeline/       the 8 scripts that produce the deliverable — run in order
├── experiments/    22 scripts: model comparison, benchmarks, negative results
├── output/         generated — gitignored
├── EDA_summary.md      chronological working log, steps 1–13
└── PROJECT_HANDOFF.md  full briefing: findings, constraints, what to do next
```

## Running the pipeline

Scripts are numbered in execution order and resolve all paths relative to
themselves, so they can be run from any working directory.

```bash
pip install -r ../requirements.txt

python pipeline/01_clean_gfp_data.py        # merge 15 xlsx -> gfp_clean.pkl + EDA plot
python pipeline/02_make_splits.py           # three leakage-checked split schemes
python pipeline/03_build_features.py        # sparse (6,559) + dense (66) matrices
python pipeline/12_derive_wt_sequences.py   # reconstruct the 4 wild-type sequences
python pipeline/16_align_backbones.py       # Needleman-Wunsch star alignment
python pipeline/17_evo_features.py          # 11 cross-homolog evolutionary features
python pipeline/21_design_variants.py       # design oracle + benchmark + beam search
python pipeline/22_make_design_report.py    # -> design_panel.csv, design.png
```

The numbering is the project's original run order (`01`…`31`), kept so that the
code inventory in `PROJECT_HANDOFF.md` and the cross-references between scripts
stay valid. The eight above still sort into the correct sequence.

The pipeline needs only the core dependencies — no PyTorch, no model downloads.
`01` is what creates `output/`; everything downstream reads its `gfp_clean.pkl`,
so run it first on a fresh clone.

**One caveat:** panel (a) of `design.png` reads `output/extrap_cv/evo_fold*.npz`,
produced by `experiments/18_evo_benchmark.py`. `design_panel.csv` itself does not
depend on it.

## What is in experiments/

The work that established *why* the pipeline looks the way it does: baselines,
an eight-family model comparison over 104 configurations, cross-validated
extrapolation benchmarks, leave-one-backbone-out transfer, and the ESM-2 arms.

It also holds the **negative results**, kept deliberately:

| Script | Question | Answer |
|---|---|---|
| `14`, `15` | Retune XGBoost for extrapolation? | No — Δ +0.0015, CI [−0.015, +0.018] |
| `20` | Two-stage classifier + regressor? | Marginal — AUC 0.778→0.791. Don't build on it |
| `23` | ESM-2 zero-shot log-odds? | Significantly worse — −0.0345 ρ, CI [−0.0511, −0.0172] |
| `25` | Wider ESM embedding block? | Flat — every CI spans zero |
| `28` | Retune the design oracle? | No — CI [−0.074, +0.223], P(>0) = 0.785 |

Two of these are load-bearing: `14` writes the paired baseline that
`18_evo_benchmark.py` compares against, and `23` builds the feature matrix
`26_esm_benchmark.py` needs for its comparison arms. Neither can be dropped
without breaking a positive result.

Only `28` and `29` import across folders (`experiments/` → `pipeline/`, for
`21_design_variants`); both put `pipeline/` on `sys.path` for that reason.

## Findings worth knowing before extending this

1. **Brightness is not comparable across libraries**, even after fold-WT
   normalisation — two byte-identical sequences read 1.000 and 0.245 in different
   libraries. Score cross-backbone work with **Spearman only**.
2. **The best model depends on the regime, and the answers are opposite.** An MLP
   wins interpolation (ρ 0.911); it loses to plain ridge at unseen positions.
   XGBoost loses the headline comparison and wins extrapolation decisively.
3. **Which features help also reverses.** The 6,559 sparse indicators are dead
   weight for transfer, and are exactly what finds the brightest recombinants.
4. **ρ ≈ 0.45 at unseen positions is dead-vs-alive separation, not brightness
   ranking.** Among functional variants it is 0.247; above 1.0× WT it is negative.
5. **Design by recombination is validated. Design by invention is not** —
   enrichment at never-assayed positions is below 1. The panel is entirely in the
   first category.

The best feature set for extrapolation is `dense + evo + ESM-2 embeddings`
(109 features): ρ 0.5185 at unseen positions, 0.553 mean on leave-one-backbone-out.
Reproducing it needs `experiments/24_esm_embed.py` and `26_esm_benchmark.py`,
which do require PyTorch and a HuggingFace download.

Even at its best, extrapolation accuracy is ρ ≈ 0.48. That is useful for triage
and for enriching a shortlist. It is not accurate enough to trust an individual
prediction.

## References

- Sarkisyan KS et al. *Local fitness landscape of the green fluorescent protein.*
  Nature 533:397–401 (2016). doi:10.1038/nature17995
- Gonzalez Somermeyer L et al. *Heterogeneity of the GFP fitness landscape and
  data-driven protein design.* eLife 11:e75842 (2022). doi:10.7554/eLife.75842
