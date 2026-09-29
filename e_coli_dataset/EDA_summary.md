# GFP dataset (E. coli) — cleaning, EDA & splits summary

## Folder layout

```
e_coli_dataset/
├── EDA_summary.md              (this file)
├── README.md                   (pipeline + experiments guide)
├── pipeline/                   the 8 scripts that produce the deliverable, in order
│   ├── 01_clean_gfp_data.py            step 1:   merge + clean + tag + EDA plot
│   ├── 02_make_splits.py               step 2:   build + verify train/val/test splits
│   ├── 03_build_features.py            step 3a:  sparse + dense feature matrices  [imported]
│   ├── 12_derive_wt_sequences.py       step 6:   reconstruct + cross-check the 4 WT sequences
│   ├── 16_align_backbones.py           step 9a:  Needleman-Wunsch star alignment  [imported]
│   ├── 17_evo_features.py              step 9b:  11 cross-homolog features  [imported]
│   ├── 21_design_variants.py           step 10:  design oracle + benchmark + beam search  [imported]
│   └── 22_make_design_report.py        step 10b: assembles design_panel.csv + design.png
├── experiments/                model comparison, benchmarks, negative results
│   ├── 04_train_baselines.py           step 3b:  ridge baselines, both splits
│   ├── 05_analyze_results.py           step 3c:  extrapolation-by-novelty breakdown + figure
│   ├── 06_compare_models.py            step 4a:  8 families x 104 configs (resume-safe)
│   ├── 08_summarize_comparison.py      step 4b:  leaderboard + comparison figure
│   ├── 07_train_mlp_full.py            step 4c:  full-data MLP, epoch checkpointing
│   ├── 09_analyze_extrapolation.py     step 4d:  novelty breakdown for the full-data MLP
│   ├── 10_cross_val_extrapolation.py   step 5a:  5-fold position CV (resume-safe, core benchmark)
│   ├── 11_analyze_extrapolation_cv.py  step 5b:  pooled folds + paired bootstrap + figure
│   ├── 13_lobo_eval.py                 step 7:   leave-one-backbone-out (resume-safe)
│   ├── 14_tune_xgb_extrapolation.py    step 8:   XGBoost re-tune for extrapolation (negative)
│   ├── 15_plot_xgb_tune.py             step 8:   figure
│   ├── 18_evo_benchmark.py             step 9c:  dense vs dense+evo, defines the folds  [imported]
│   ├── 19_evo_lobo.py                  step 9d:  dense vs dense+evo, cross-protein
│   ├── 20_two_stage_benchmark.py       step 10a: classifier + functional-only regressor
│   ├── 23_esm_scores.py                step 11a: ESM-2 zero-shot masked-marginals (needs weights)
│   ├── 24_esm_embed.py                 step 11b: ESM-2 per-residue embeddings, PCA  [imported]
│   ├── 25_esm_dims_sweep.py            step 11b: PCA width sweep (flat; 16 kept)
│   ├── 26_esm_benchmark.py             step 11:  all ESM arms on the 5-fold CV + bootstraps
│   ├── 27_lobo_esm.py                  step 11c: leave-one-backbone-out with ESM arms
│   ├── 28_design_optimize.py           step 12:  design-oracle re-tune (negative, kept as-is)
│   ├── 29_design_stress_test.py        step 13:  design benchmark split by near-neighbour
│   ├── 30_make_esm_report.py           step 11-12: assembles esm.png
│   └── 31_make_slides.js               deck:     regenerates GFP_ML_results.pptx (pptxgenjs)
│       [imported] = also loaded as a module by a later script;
│       renumbering one means updating the importlib string that names it
│       28 and 29 import 21 from pipeline/ and add it to sys.path
├── dataset/                    15 source GFP_variants_part*.xlsx files + column docs
└── output/
    ├── gfp_clean.pkl           intermediate cleaned dataframe
    ├── gfp_variants_clean.csv  cleaned, modeling-ready dataset
    ├── gfp_variants_split.csv  cleaned data + split columns
    ├── features/               X_sparse.npz, X_dense.npy, y.npy, meta.csv
    ├── metrics_random.csv      baseline metrics, random split
    ├── metrics_position.csv    baseline metrics, position-holdout split
    ├── extrapolation_breakdown.csv
    ├── model_comparison.csv    every config evaluated (104 rows)
    ├── model_leaderboard.csv   best config per model/tier/split
    ├── eda_distributions.png   brightness & mutation-count histograms
    ├── baseline_results.png    baseline comparison + error analysis
    └── model_comparison.png    8-model leaderboard figure
```

## Step 1: what was done

1. Merged all 15 `GFP_variants_part*.xlsx` files → 141,469 rows.
2. Dropped 3 sequences with unresolved `X` residues (natural FPs cgigGFP, scubGFP1, scubGFP2).
3. Dropped 316 rows with no usable label (blank `Brightness value`) — this removes all 49 standalone natural-FP wild-types (no same-family WT to normalize against) and all 264 ML-designed multi-mutants (colony-microscopy assay has no published WT reference point to convert to fold-WT).
4. Tagged every remaining row with `Backbone` (avGFP / amacGFP / cgreGFP / ppluGFP2) and `Source group` (DMS vs. engineered/classic), based on the assay/analyzing method (not the classification label — several "Wild-type" rows are actually on the same FACS-seq scale as the DMS data, so tagging by method rather than label keeps them correctly grouped).
5. Dropped the `% positive cells` column — confirmed 100% empty after cleaning, matching the known gap in the README.

**Result: 141,150 rows, 12 columns**, saved as `output/gfp_variants_clean.csv`.

## Key findings

**Backbone counts (DMS rows):** avGFP 51,715 · amacGFP 33,511 · ppluGFP2 31,402 · cgreGFP 24,516 (141,144 total). Each backbone has a single fixed sequence length (no indels), so alignment-free, position-wise mutation encoding is straightforward.

**Brightness distribution is bimodal** (see `output/eda_distributions.png`) — a large spike near 0 (loss-of-function variants) and a peak near 1.0 (near-WT brightness), typical of a protein fitness landscape. Worth considering a two-part model (bright/dark classifier + regression on the bright subset) or a zero-inflated loss, not just plain regression.

**MFI and Brightness value are perfectly correlated within each backbone (r=1.0)** — brightness is just a linear rescale of MFI by that backbone's WT value, confirming the README's guidance that brightness (not raw MFI) is the right label for pooling across backbones.

**Important data-quality catch:** 6 surviving "engineered/classic" avGFP-lineage rows (EGFP, sfGFP, Emerald, mEGFP, GFP S65T, avGFP wild-type reference) have brightness values on a completely different scale (19.8–54.1) than the avGFP DMS rows (0–2.5), because they were measured by molecular brightness (EC×QY) rather than FACS-seq MFI ratio. They are tagged `Source group = engineered/classic` specifically so they can be excluded or modeled separately — pooling them with the DMS rows as-is would corrupt the label scale. (Note: 4 other "Wild-type" rows — the FACS-seq wild-type reference points for cgreGFP, ppluGFP2, amacGFP, and the avGFP F64L parent — are correctly kept in the `DMS` group since they're on the same scale, despite also being labeled "Wild-type".)

**Mutation counts** per variant range from 0 (WT) to 43, with median 2–3 mutations across all backbones — consistent with the DMS libraries being low-order combinatorial mutants.

## Step 2: train/val/test splits (`output/gfp_variants_split.csv`, built by `pipeline/02_make_splits.py`)

Three independent split schemes were added as columns on top of the cleaned data, all excluding the 6 engineered/classic rows:

- `split_random`: stratified by (backbone, dark/bright), group-aware so duplicate sequences stay together — 112,914 / 14,115 / 14,115 train/val/test. This is the optimistic upper-bound baseline.
- `split_position_holdout`: per-backbone, entire mutated positions held out (train never sees a mutation at a held-out position) — 83,986 / 27,609 / 29,549 train/val/test. This is the realistic extrapolation test.
- `split_backbone_holdout_eligible`: flags rows usable for leave-one-backbone-out cross-validation (train on 3 backbones, test on the 4th).

Why this matters: the DMS libraries are combinatorial, so most variants differ from wild-type by only 2–3 mutations and share nearly all their sequence with other rows. A plain random split scatters near-identical sequences across train and test, letting a model score well by memorization. The position-holdout split is the honest test of whether the model learned anything transferable.

Verified: zero train rows touch a held-out position in any backbone, and the random split has zero duplicate-sequence leakage across train/val/test. (Two rows show up as "leaking" duplicates in the position-holdout split, but they're a real biological curiosity, not a bug: an avGFP wild-type sequence is byte-identical to a heavily "humanized" 38-mutation amacGFP variant — convergent sequences across two different backbones, which position-holdout can't detect since it partitions positions per-backbone.)

## Step 3: baseline models (`pipeline/03_build_features.py`, `04_train_baselines.py`, `05_analyze_results.py`)

**Features.** Two blocks, 141,144 rows:

- *Sparse* (6,559 cols): one binary column per `(backbone, position, mutant AA)`. Keying on backbone makes a linear fit equivalent to four per-backbone models sharing a regulariser — deliberate, given brightness is not comparable across libraries.
- *Dense* (66 cols): features that still mean something at an unseen position — BLOSUM62 scores, hydropathy/volume/charge deltas, mutated-from/to amino-acid counts, mutation count, backbone one-hot.

Mutation parsing was validated against the sequences: all ~48k checked mutations match `sequence[position-1]`, confirming clean 1-indexing.

**Test-set Spearman ρ:**

| model | random split | position-holdout |
|---|---|---|
| mean_by_backbone (floor) | 0.303 | 0.366 |
| ridge_sparse | 0.878 | 0.792 |
| ridge_dense | 0.610 | 0.666 |
| **ridge_combined** | **0.880** | **0.814** |
| hgb_dense | 0.776 | 0.745 |

R² for `ridge_combined`: 0.705 (random), 0.552 (position-holdout).

**The headline position-holdout number is optimistic, and the breakdown shows why.** Most test rows are multi-mutants carrying just *one* mutation at a held-out position while their remaining mutations sit at positions the model trained on. Isolating genuine extrapolation:

| test subset | n | ridge_sparse | ridge_dense | ridge_combined |
|---|---|---|---|---|
| all test rows | 29,549 | 0.792 | 0.666 | 0.814 |
| 1 novel mutation + others seen | 20,567 | 0.816 | 0.641 | 0.827 |
| every mutation novel | 1,086 | 0.332 | 0.475 | 0.455 |
| single mutant (pure extrapolation) | 293 | **0.268** | 0.387 | 0.377 |

On pure extrapolation `ridge_sparse` scores **exactly** the mean-predictor floor (0.268 both) — as it must, since a never-seen position has no fitted coefficient, so the model returns the backbone intercept. The transferable dense features are the only thing carrying signal there (0.387 vs 0.268), and they're worth ~0.12 ρ. That is the real measure of how far these baselines generalize: **useful for ranking recombinations of known mutations, close to guessing for genuinely new sites.**

**Second limitation** (middle panel of `output/baseline_results.png`): the model badly under-separates the dark mode, predicting ~0.5–0.75 for variants whose true brightness is ~0. An additive model cannot express "one bad mutation kills the protein regardless of the rest," which is exactly what the dark spike is.

## Step 4: eight-model comparison (`experiments/06_compare_models.py`, `08_summarize_comparison.py`)

104 configurations across 8 model families, all on identical features and identical test sets. Hyperparameters selected on the **validation** split by Spearman ρ; test is never used for selection.

Two tiers, because the families don't scale alike:

- **full** — whole training split (112,914 rows random / 83,986 position). Only models that scale.
- **sub12k** — a fixed 12,000-row subsample of the *same* training split, scored on the *same* full val/test sets. Includes every model, so all 8 are directly comparable.

**Test Spearman ρ, best config per model:**

| model | random / full | random / sub12k | position / full | position / sub12k |
|---|---|---|---|---|
| **mlp** | **0.911** | 0.848 | **0.823** | 0.791 |
| linear_svr | 0.882 | 0.830 | 0.800 | 0.787 |
| ridge | 0.880 | 0.829 | 0.814 | 0.780 |
| lasso | 0.854 | 0.724 | 0.793 | 0.729 |
| elastic_net | 0.854 | 0.724 | 0.793 | 0.729 |
| xgboost | 0.813 | 0.774 | 0.786 | 0.759 |
| random_forest | — | 0.681 | — | 0.695 |
| svr_rbf | — | 0.666 | — | 0.661 |

Test R² tells the same story even more strongly on the random split: MLP **0.870** vs ridge 0.705.

### What the comparison shows

**Linear models win, and that is a result about the biology, not a shortcut.** Ridge and LinearSVR top the full tier (ρ ≈ 0.88). The mutation→brightness landscape is largely *additive* — individual mutation effects roughly sum — and the sparse one-hot encoding makes exactly that structure trivial for a linear model to fit.

**XGBoost loses to plain ridge** (0.813 vs 0.880). Trees are a poor match for 6,625 sparse binary columns where only ~3 are active per row: each split isolates a tiny subset of variants, so the model spends capacity rediscovering additive effects a linear term captures in one coefficient.

**L1 is actively harmful.** Lasso and ElasticNet trail ridge by ~0.03 ρ, and their best configs sit at the smallest penalty tested (α=1e-6) — i.e. the search wanted *as little L1 as possible*. Most mutations contribute a little signal, so zeroing coefficients discards real information.

**The MLP wins outright once trained on the full data** (`experiments/07_train_mlp_full.py`). It reaches ρ=0.911 / R²=0.870 on the random split and ρ=0.823 on position-holdout, beating ridge on both. The prediction made from the subsample tier held: MLP led at equal data (0.848 vs 0.829), and scaling to all 113k rows added a further ~0.06 ρ.

The gap is widest in R² (0.870 vs 0.705), which says the MLP is not merely ranking variants better but predicting their actual brightness far more accurately — consistent with it being the only model in the comparison that can represent the non-additive "one bad mutation kills the protein" behaviour driving the dark mode.

**Random forest and kernel SVR are the weakest** (ρ ≈ 0.66–0.70), both struggling with high-dimensional sparse binary input.

### Compute caveats (these constrained the search)

- **RandomForest and kernel SVR could not train on the full data** — hence the sub12k tier. The MLP was subsequently trained at full scale by `07_train_mlp_full.py`, which drives epochs manually with `partial_fit`, scores the real validation split after each one, keeps the best epoch's weights, and checkpoints to disk — so the ~9-minute fit can span multiple runs. Both full-data MLPs stopped early (best epoch 6 of 10). Its `alpha` was carried over from the sub12k search rather than re-tuned at full scale.
- **Kernel SVR** additionally used only 6,000 training rows (recorded in `n_train`), and its search was capped at `gamma='scale'`, `C ≤ 1`. Larger values explode the support-vector count past the wall-clock limit. Its numbers are therefore a *lower bound*.
- **Lasso/ElasticNet use SGD**, not exact coordinate descent — exact Lasso needs ~103s per fit on this matrix. Same penalty, approximate solver.
- **HistGradientBoosting was excluded** — it cannot accept sparse input, and densifying 113k × 6,625 needs ~3 GB. XGBoost represents gradient boosting here.

## Step 4d: does the MLP's advantage survive real extrapolation? (`experiments/09_analyze_extrapolation.py`)

No — it disappears, and the ordering inverts. Position-holdout test set, Spearman ρ:

| test subset | n | floor | ridge_combined | xgboost | mlp_full |
|---|---|---|---|---|---|
| all test rows | 29,549 | 0.366 | 0.814 | 0.786 | **0.823** |
| 1 novel + others seen | 20,567 | 0.320 | 0.827 | 0.788 | **0.831** |
| every mutation novel | 1,086 | 0.332 | 0.455 | **0.459** | 0.416 |
| single mutant (pure extrapolation) | 293 | 0.268 | 0.377 | **0.443** | 0.372 |

(Pipeline self-check: the reconstructed MLP overall ρ reproduces 0.8228 exactly.)

**The MLP's entire advantage comes from recombining mutations it has already seen.** It leads by 0.04 ρ on rows where the novel mutation is cushioned by familiar ones, and is *behind* ridge on single mutants at never-seen positions (0.372 vs 0.377). Its strength is interpolation within known mutation space, not generalisation to new sites.

**XGBoost is the best extrapolator**, despite finishing last overall — ρ=0.443 on pure extrapolation versus ~0.375 for both ridge and the MLP. A paired bootstrap over the 293 single-mutant rows (2,000 resamples) puts the advantage at Δρ=+0.070 vs MLP (95% CI [−0.003, +0.144], P(Δ>0)=0.97) and +0.064 vs ridge (95% CI [−0.004, +0.130], P=0.97). **Suggestive, not established** — both intervals barely cross zero, and 293 rows is thin. It needs more single-mutant test data before being treated as fact.

The practical implication: **model choice should depend on the use case.** For ranking recombinations of characterised mutations, the MLP is clearly best. For scoring mutations at positions never assayed — the harder and more valuable design problem — no model here does much better than ρ≈0.44, and the leader is the one that looked worst on the headline metric.

## Step 5: the extrapolation benchmark, settled (`experiments/10_cross_val_extrapolation.py`, `11_analyze_extrapolation_cv.py`)

Step 4d's finding rested on 293 single mutants with a CI that crossed zero. The dataset actually contains **4,596** single mutants — the old benchmark was small only because the position holdout withheld 6% of positions, so only 6% of them landed in test.

**5-fold cross-validation over positions** fixes that. Positions are split into 5 folds per backbone; each fold trains on rows whose mutations all sit *outside* the fold and tests on rows whose mutations all sit *inside* it. Rows straddling the boundary are dropped from both, so no training row ever touches a held-out position. Every variant is tested exactly once by a model that never saw any of its positions — **15.7× more evidence from the same data**.

**Out-of-fold Spearman ρ at never-seen positions:**

| subset | n | ridge | **xgboost** | mlp |
|---|---|---|---|---|
| all-novel variants | 16,017 | 0.367 | **0.422** | 0.339 |
| **single mutants** | **4,596** | 0.368 | **0.441** | 0.329 |
| single — amacGFP | 1,201 | 0.251 | **0.384** | 0.177 |
| single — avGFP | 1,085 | 0.353 | **0.442** | 0.297 |
| single — cgreGFP | 1,169 | 0.395 | **0.496** | 0.425 |
| single — ppluGFP2 | 1,141 | 0.299 | **0.301** | 0.284 |

**Paired bootstrap (4,000 resamples, 4,596 rows) — all three differences are now significant:**

| comparison | Δρ | 95% CI | verdict |
|---|---|---|---|
| xgboost − mlp | +0.112 | [+0.091, +0.133] | significant |
| xgboost − ridge | +0.072 | [+0.054, +0.091] | significant |
| ridge − mlp | +0.039 | [+0.017, +0.061] | significant |

**The step-4d lead is confirmed and strengthened.** XGBoost is decisively the best extrapolator, and it wins in all four backbones. More striking: **the MLP is significantly *worse* than plain ridge** at unseen positions — the model that leads the headline benchmark by 0.03 ρ is last where it matters for designing novel variants, trailing XGBoost by 0.11.

The reading is that the MLP's extra capacity is spent learning the specific mutation landscape it was shown, which is exactly what fails to transfer. XGBoost's shallow-interaction structure generalises better to sites it has never seen. ppluGFP2 is the one backbone where the models tie, and it is also the one where every model does worst.

## Step 6: ESM-2 scoring pipeline — written, waiting on weights (`experiments/23_esm_scores.py`)

Ready to run offline the moment an ESM-2 checkpoint and a Linux CPU PyTorch wheel are dropped into the project folder (this sandbox reaches PyPI only; both model-weight hosts and PyTorch's CPU index are blocked by its egress proxy).

The approach is the standard zero-shot formulation (Meier et al. 2021) and is cheap: `score = log P(mut at p) − log P(wt at p)`, needing only ~L forward passes **per backbone**, not per variant. The four wild-type sequences are queried once each to build an (L × 20) log-probability matrix, then all 141,150 variants are scored by table lookup.

Supporting work done and verified:

- **Wild-type sequences derived and cross-checked** (`12_derive_wt_sequences.py`). Each backbone's WT was reconstructed by reverting every variant's own mutations independently: **141,141 of 141,142 reversions agree**. The single dissenter is `avGFP (parent, F64L)`, whose name parses `F64L` as a mutation — confirming the DMS library's reference is the **F64L parent** (L at position 64), which is what the mutation numbering is relative to.
- **Pipeline dry-run validated**: with a random log-probability matrix, 100% of variants score correctly and the residue-consistency check catches exactly the one expected mismatch. Only the ~25 lines that call torch remain untested.

## Step 7: leave-one-backbone-out (`experiments/13_lobo_eval.py`)

The hardest generalisation test, and the last item pending from step 2: train on three backbones, predict the fourth. Scored with **Spearman only** — fold-WT brightness is not comparable across libraries (step 1).

| features | model | amacGFP | avGFP | cgreGFP | ppluGFP2 | **mean** |
|---|---|---|---|---|---|---|
| dense (66) | ridge | 0.308 | 0.609 | 0.544 | 0.354 | 0.454 |
| dense (66) | **xgboost** | 0.421 | 0.623 | 0.539 | 0.339 | **0.481** |
| dense (66) | mlp | 0.318 | 0.559 | 0.552 | 0.352 | 0.445 |
| combined (6,625) | ridge | 0.293 | 0.624 | 0.544 | 0.336 | 0.449 |
| combined (6,625) | **xgboost** | 0.365 | 0.635 | 0.566 | 0.365 | **0.483** |

**Three findings:**

1. **XGBoost wins again** — best mean in both feature blocks. That is now the *third independent* confirmation that trees generalise best here (position-holdout, the cross-validated single-mutant benchmark, and now cross-protein). The MLP is again last.
2. **The 6,559 sparse indicators contribute essentially nothing** (ridge 0.454 dense vs 0.449 combined; xgboost 0.481 vs 0.483). Exactly as the construction predicts: for a held-out protein, every one of its position columns is zero throughout training. **99% of the feature space is dead weight in this regime** — the 66 transferable descriptors carry all the signal.
3. **Cross-protein transfer is better than expected.** Mean ρ ≈ 0.48 for predicting an entirely different protein is on par with the 0.44 for predicting an unseen *site within a known* protein. Predicting a new GFP is about as hard as predicting a new position — not dramatically harder.

Per-backbone variance is large though: avGFP is easiest (ρ 0.62), ppluGFP2 and amacGFP hardest (0.34–0.42). avGFP being the easiest to predict from the others is consistent with the step-1 duplicate finding, where an amacGFP variant converged to an exactly avGFP sequence — those two backbones are related, so training on one genuinely informs the other.

## Step 8: retuning XGBoost for extrapolation — a negative result (`experiments/14_tune_xgb_extrapolation.py`)

Every XGBoost number so far used one fixed config (`n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8`) that was never actually chosen for this job — it came from the step-4 grid, which picked configs by validation Spearman on the **random** split, the one regime where XGBoost is weakest. Two hypotheses motivated a proper retune: (a) a config chosen for the extrapolation objective should do better than one chosen for interpolation, and (b) per step 7, the 6,559 sparse position indicators are dead weight once positions are unseen, so tuning can drop them and use the 66 dense descriptors only — cheaper fits, same fold definition as step 5 (`SEED=7`, `assign_position_folds`) so results are directly comparable.

**Phase 1 — cheap screen.** A 48-config grid (`max_depth` 3/4/6/8, `learning_rate` 0.03/0.05/0.1, `n_estimators` 300/600, `min_child_weight` 1/5) run on 2 of the 5 folds, scored on each fold's single-mutant rows. Shallower, more regularised trees looked clearly better than the default: top config (depth 4, lr 0.03, 300 trees) averaged ρ=0.461 across those two folds versus ≈0.42 for a default-like config.

**Phase 2 — the real test.** The top 3 candidates plus the default were run across **all 5 folds** and pooled (n=4,596 single mutants, same set used throughout step 5), then paired-bootstrapped (4,000 resamples) against the default:

| config | max_depth | learning_rate | n_estimators | pooled ρ (single) | Δ vs default | 95% CI | P(Δ>0) |
|---|---|---|---|---|---|---|---|
| default | 8 | 0.10 | 400 | 0.4377 | — | — | — |
| cand_8 | 3 | 0.10 | 300 | 0.4393 | +0.0015 | [−0.0145, +0.0175] | 0.563 |
| cand_12 | 4 | 0.03 | 300 | 0.4362 | −0.0012 | [−0.0193, +0.0168] | 0.449 |
| cand_13 | 4 | 0.03 | 300 | 0.4356 | −0.0020 | [−0.0195, +0.0157] | 0.413 |

**The apparent gain from phase 1 did not replicate.** Pooled across all 5 folds, every candidate is statistically indistinguishable from the untuned default — all three CIs straddle zero comfortably, none of the P(Δ>0) values are near the 0.95+ threshold used everywhere else in this project. The phase-1 "win" was 2-fold sampling noise, not a real effect. **Conclusion: the default XGBoost config, despite being selected for a different objective, is already about as good as anything nearby in this search space for extrapolation** — there is no free improvement sitting in `max_depth`/`learning_rate`/`n_estimators`/`min_child_weight`. A wider or more exotic search (different feature subsets, monotonicity constraints, explicit position-frequency features) would be needed to move this number, not more grid search in this neighbourhood.

**A useful confirmation fell out of the same run.** The default config trained on dense-only features (no sparse block) scored ρ=0.4377 on pooled single mutants — statistically the same as step 5's combined-feature result (ρ=0.441, same fold definition, same 4,596 rows). This directly confirms step 7's finding on the harder within-protein extrapolation task, not just the cross-protein one: **the 6,559 sparse indicators can be dropped for any novel-position task at no measurable cost**, and dense-only fits ran roughly 2–4× faster in this run (2.5–8s vs the combined block's heavier per-fit cost in step 5).

## Step 9: outside-the-assay information at last breaks the plateau (`pipeline/16_align_backbones.py`, `17_evo_features.py`, `18_evo_benchmark.py`, `19_evo_lobo.py`)

Every model has plateaued near ρ≈0.44 at never-assayed positions for one reason: nothing in the 66 descriptors says anything about a *specific site*. BLOSUM62 knows how often Leu replaces Ile across all proteins; it does not know that position 66 is the chromophore tyrosine and must never change. Step 8 closed off hyperparameter tuning as a route to fixing that. The only remaining route is to import site-specific knowledge from outside this assay.

ESM-2 was the intended source and remains unreachable offline (rechecked: `pypi.org` returns 200, `huggingface.co` / `dl.fbaipublicfiles.com` / `download.pytorch.org` all return 000 through the sandbox proxy; `torch` is still not installed). So step 9 used **the outside information that is already sitting in the dataset**: the four backbones are homologous GFPs from four different organisms, so the differences between them are themselves an evolutionary record of what a GFP tolerates at each site.

### 9a. Aligning the four wild-types (`16_align_backbones.py`)

The sequences differ in length (238/238/235/222), so they need aligning rather than indexing. Implemented Needleman–Wunsch with BLOSUM62 and affine gaps (Gotoh, gap open −11 / extend −1) in a star topology against avGFP; 255 alignment columns, every residue of every backbone mapped.

**Two independent validations passed:**

- **The chromophore aligns.** avGFP Tyr66-Gly67 lands in the same two columns in all four backbones — the one motif that must be invariant across the GFP family. (Position 65 reads S, not T, correctly reflecting that this dataset's reference is the F64L parent rather than the S65T-containing EGFP lineage.)
- **Pairwise identities match the known phylogeny:** amacGFP–avGFP 82.8%, cgreGFP ~44–45% to both, and ppluGFP2 the distant outlier at 17.8–24.9%. The amacGFP–avGFP closeness independently corroborates the step-1 duplicate finding, where an amacGFP variant converged to an exactly avGFP sequence.

### 9b. Eleven cross-homolog features (`17_evo_features.py`)

For a mutation wt→mut at position *p* on backbone *B*, with *H* = the residues of the other three backbones at the aligned column: `conservation` (fraction of *H* equal to wt), `mut_in_homolog` (is this exact substitution one nature already made?), and `evo_score` = mean BLOSUM62(mut, *H*) − mean BLOSUM62(wt, *H*) — deliberately the same log-odds shape as the ESM score, with a three-sequence column standing in for a language model. Aggregated per variant into 11 features. 100% of variants scored, with exactly one residue mismatch flagged — the same known F64L parent row that `23_esm_scores.py`'s dry run catches.

**No label information is used**: the features derive only from the four wild-type sequences, so they are constant per (backbone, position, mutant) and cannot leak brightness across a fold.

### 9c. The benchmark (`18_evo_benchmark.py`)

Identical folds to step 5 (`SEED=7`), identical XGBoost config (the one step 8 confirmed), identical pooled single-mutant metric. The dense-only baseline is not recomputed — step 8 already wrote exactly this model on exactly these folds, so the comparison is *exactly paired*.

| subset | n | dense (66) | dense+evo (77) | Δ |
|---|---|---|---|---|
| all-novel variants | 16,017 | 0.4060 | **0.4554** | **+0.049** |
| **single mutants** | **4,596** | 0.4377 | **0.4822** | **+0.044** |
| single — amacGFP | 1,201 | 0.4058 | 0.4330 | +0.027 |
| single — avGFP | 1,085 | 0.4874 | 0.5391 | +0.052 |
| single — cgreGFP | 1,169 | 0.4763 | **0.6251** | **+0.149** |
| single — ppluGFP2 | 1,141 | 0.2879 | 0.2567 | −0.031 |

**Paired bootstrap (4,000 resamples): single mutants Δρ=+0.0444, 95% CI [+0.0275, +0.0603], P(Δ>0)=1.000 — significant.** All-novel variants Δρ=+0.0494, CI [+0.0405, +0.0583], also significant.

**The model genuinely uses them**: `evo_score_sum` is the single highest-gain feature in the whole model, above every one of the 66 dense descriptors, and the evo block carries 17.7% of total importance on 14.3% of the width.

### 9d. And on an entirely new protein (`19_evo_lobo.py`)

Leave-one-backbone-out, same protocol as step 7. (Fair by construction: for a held-out backbone the evo features come from the other three wild-type *sequences* — if you are engineering a new FP you know its sequence, and no brightness measurement from the held-out protein is used.)

| features | amacGFP | avGFP | cgreGFP | ppluGFP2 | **mean** |
|---|---|---|---|---|---|
| dense (66) | 0.4209 | 0.6227 | 0.5391 | 0.3392 | 0.4805 |
| dense+evo (77) | **0.4853** | **0.6750** | **0.6181** | 0.3173 | **0.5239** |
| Δ | +0.064 | +0.052 | +0.079 | −0.022 | **+0.043** |

### What this means

**This is the first change in the whole project that has moved the extrapolation number.** Steps 4–8 reshuffled models and hyperparameters within one fixed view of a mutation and never beat ρ≈0.44; adding genuine site-specific information beat it immediately, in both the within-protein (+0.044) and cross-protein (+0.043) settings, and it did so from eleven features computed off four sequences with no network access.

**ppluGFP2 is the informative exception, and it fails in exactly the predicted direction.** It is the one backbone that degrades, in *both* experiments (−0.031 within-protein, −0.022 cross-protein). It is also the phylogenetic outlier at 17.8–24.9% identity. At that distance the aligned columns carry little real constraint — shared residues are as likely to be alignment artefact as conservation — so the features are noise for it. Conversely cgreGFP gains most (+0.149, +0.079), and it is the backbone with two homologs at ~44–45% identity: close enough to align confidently, distant enough that a shared residue means evolution actually held it fixed. **The benefit tracks having relatives at intermediate evolutionary distance**, which is precisely the regime a real MSA or a protein language model would exploit far better than three sequences can.

That last point is the strongest argument yet for the ESM-2 experiment rather than a replacement for it: if three sequences are worth +0.044, the millions distilled into a PLM should be worth considerably more — and would not have the ppluGFP2 blind spot, since ESM-2 has seen copepod FPs that this dataset's other three backbones know nothing about.

## Recommendation for next step

1. **Protein language model embeddings (ESM-2)** — still the highest-value remaining experiment, and step 9 has now made the case empirically rather than by assertion: site-specific evolutionary information is worth +0.044 ρ even in its crudest three-sequence form, and a PLM is the same idea with millions of sequences behind it and no outlier blind spot. Step 8 closed off the cheap route to improving extrapolation (hyperparameter tuning); nothing in the current feature set describes a site the assay never touched, and a PLM is the only candidate that brings outside evolutionary information rather than re-slicing the same 66 descriptors. Pair it with dense-only XGBoost using the (now confirmed) default config.

   *Not runnable in this sandbox*: only PyPI is reachable. Both ESM weight sources (`dl.fbaipublicfiles.com` and `huggingface.co`) are blocked, and `torch` resolves to a 3 GB CUDA build with its CPU-only index proxy-blocked. This needs an environment with general internet access. The cheap route once available is the masked-marginals score — one forward pass per backbone gives a (length × 20) log-probability matrix, so a mutation's score is `log P(mut) − log P(wt)`; no per-variant inference needed.
2. **Tune the MLP properly at full scale** — architecture, alpha, and learning rate were all selected on the 12k subsample. The full-data run stopped at epoch 6 of 10, so a proper schedule (learning-rate decay, more epochs, wider nets) likely has more to give for the interpolation/recombination use case, where the MLP is already the leader.
3. **Two-stage model** — dead/alive classifier followed by a regressor on the bright subset. Still worth testing, though the MLP's R² jump to 0.870 suggests it already captures much of the bimodality that defeated the additive models.
4. Consider log-transforming brightness for the regression stage (rank metrics unaffected, RMSE/R² are).
5. **Settled by step 8**: use dense-only features (drop the 6,559 sparse indicators) for any extrapolation/transfer model — confirmed at no measurable cost on both the cross-protein (step 7) and within-protein unseen-position (step 8) tasks, with a meaningful speed gain.

---

## Step 10 — design / optimisation (the second half of the objective)

Steps 1-9 built a predictor. This step uses it to propose sequences, which was
the remaining half of the project goal and had no code at all.

### 10.0 The obvious design loop does not work — check before trusting

Before maximising the step-9 model over candidate sequences, its out-of-fold
predictions were decomposed by brightness stratum. The ~0.45 Spearman turns out
to be almost entirely dead-vs-alive separation, not brightness ranking:

| subset | n | Spearman |
|---|---|---|
| all novel variants | 16,017 | 0.455 |
| functional (y >= 0.5) | 12,141 | 0.247 |
| y >= 0.8 | 8,734 | 0.130 |
| **y >= 1.0** | **2,698** | **-0.146** |

AUC for "is this variant alive" is 0.778. Variants with y >= 1.2 sit at median
predicted rank 6,452 of 16,017 — the middle of the pack. Predictions are
compressed to a 1.29 ceiling while truth reaches 2.48 (13 predictions exceed
1.2x where 505 real variants do). Precision@20 for y >= 1.2 is 0.000 against a
3.2% base rate, i.e. **enrichment below 1**.

At never-assayed positions the model is a foldability filter. Maximising it in
the bright regime optimises a quantity uncorrelated with truth.

### 10a Two-stage model (`20_two_stage_benchmark.py`) — closes open item #3

Whether this was fixable by splitting the problem, on the settled 5-fold
position CV (folds imported from `evo_benchmark` so they are identical):

| question | model | metric | value |
|---|---|---|---|
| dead/alive | step-9 regressor | AUC | 0.778 |
| dead/alive | dedicated classifier | AUC | **0.791** |
| y >= 0.5 | step-9 regressor | rho | 0.247 |
| y >= 0.5 | regressor on functional rows only | rho | **0.262** |
| y >= 0.8 | " | rho | 0.130 -> **0.165** |
| y >= 1.0 | " | rho | -0.146 -> **-0.100** |
| precision@100, y>=1.2 | all three variants | frac | 0.02-0.03 vs 0.032 base |

Marginal gains, verdict unchanged: no model here enriches for brightness above
functional at unseen positions. **Open item #3 is answered — worth having, not
worth building the design step on.**

### 10b It is not a data ceiling — it is a transfer failure

Same features, same config, random split (interpolation): overall rho 0.820,
AUC alive 0.935, and **precision@20 for y >= 1.2 = 1.00 against a 3.0% base
rate**. The bright signal is real and learnable. What fails is transfer to
never-assayed sites — exactly the step-9 diagnosis.

**Consequence: design by recombination, not by invention.** Propose novel
combinations of substitutions that have each already been individually
characterised in that backbone's library. That places the design in the regime
where the model works, and it is directly measurable.

### 10c The design-regime benchmark (`21_design_variants.py --phase benchmark`)

Evaluation population = test rows with >= 2 mutations where every substitution
appears in training and the combination does not: 13,581 rows, mean 3.35
mutations. Precision@20:

| threshold | base rate | dense+evo (77) | dense+evo+sparse (6,636) |
|---|---|---|---|
| y >= 1.0 | 0.111 | 1.00 | 0.95 |
| y >= 1.2 | 0.030 | 0.95 | 0.80 |
| **y >= 1.5** | **0.005** | **0.15** | **0.45** |

**A reversal worth recording.** The sparse block was retired in steps 4/7/8 as
dead weight — correctly, *for extrapolation*. For recombination it is exactly
the memorised per-substitution knowledge the task exploits. Paired bootstrap on
precision@20 at y >= 1.5: **sparse - dense = +0.372, 95% CI [+0.100, +0.650],
P(>0) = 0.998.** Combining the two models dilutes the high-brightness tail, so
the design oracle uses dense+evo+sparse alone — the only model in this project
that does.

### 10d The constraint that makes stacking work

Beam search initially returned 6-mutation designs whose predictions *rose* with
depth (1.431 -> 1.763) while the empirical base rate *falls* ~28x. That is the
search exploiting model optimism. Checking the matched population resolves it —
avGFP multi-mutants where every component was assayed alone:

| components | 2 mut | 3 mut | 4 mut | 5 mut | 6 mut |
|---|---|---|---|---|---|
| every component >= 1.0x WT | 1.078 | 1.103 | 1.074 | — | — |
| every component >= 0.9x WT | 0.991 | 0.982 | 0.944 | 0.937 | 0.914 |
| unfiltered | 0.701 | 0.545 | 0.365 | 0.217 | 0.118 |

(mean brightness, fold WT)

**The collapse of brightness with mutation count is caused by accumulating
deleterious mutations, not by stacking as such.** Conditioned on good
components, brightness is roughly preserved to 5-6 mutations. The design loop
therefore restricts its extension pool to substitutions individually measured
at >= 0.95x WT (80 of 1,778 characterised for avGFP), and caps depth at 5.

A second point: the oracle's single-mutant predictions are heavily compressed
(K158G measures 2.48 and is predicted 0.917, rank 251). Where a direct
measurement exists it is the better ground truth for a *component*; the oracle's
validated job is ranking *combinations*.

### 10e Positive control — the loop rediscovers superfolder GFP

No literature input enters the pipeline. Ranking all 1,778 characterised avGFP
substitutions by the oracle puts **V163A rank 1 (cycle-3), Y39N rank 2
(superfolder), I171V rank 3 (superfolder), Y145F rank 50 (superfolder)** — four
of the eight literature mutations present in the library inside the top 2.8%.
The top designs are dominated by Y39N / V163A / I171V / Y145F / position 105,
i.e. the loop reconstructs the superfolder GFP mutation set from the DMS data
alone. (F99S 550, M153T 371, N105T 440 and A206V 889 rank mid-pack — A206V is a
genuine miss at measured 1.37. S65T and S30R are not in the avGFP library.)

### 10f Deliverable

`output/design_panel.csv` — 23 avGFP variants: 5 designs at each of 2, 3, 4 and
5 mutations plus 3 single-mutant controls. Every component individually
measured at >= 0.95x WT; **none of the designs exist in the library** (0 of the
top 50 are known variants). Each row carries predicted brightness, seed-ensemble
sd, weakest component, and the empirical P(y >= 1.2x) for its depth given
screened components (13-17%).

Expected yield from 10c, for a 20-variant plate in this regime: **~19 at
>= 1.2x WT and ~9 at >= 1.5x WT** by the sparse oracle's precision@20.

Files: `21_design_variants.py`, `20_two_stage_benchmark.py`, `22_make_design_report.py`;
`design_panel.csv`, `design_shortlist.csv` (all 38,000+ scored designs),
`design_singles.csv`, `design_regime_benchmark.csv`, `design_regime_bootstrap.csv`,
`two_stage_results.csv`, `design_enrichment.csv`, `design_oracle.joblib`,
`design.png`.

**Caveat to carry.** The recombination oracle is validated on *random-split*
held-out combinations, which is the correct regime for this task but is the
optimistic end of this project's evaluations. Nothing here licenses proposing
mutations at never-assayed positions — 10.0 and 10a show that fails.

---

## Step 11 — ESM-2, finally run (the project's #1 blocked item)

### 11.0 How it got unblocked

`dl.fbaipublicfiles.com` and `download.pytorch.org` are still 403. **`huggingface.co`
returns 200**, so the weights came by a different route. `23_esm_scores.py` gained an
`--hf-model` backend; the masked-marginals formulation, the token offset and the
wild-type residue checks are unchanged, and the original `--checkpoint` path still
works if a local `.pt` ever appears. Resume-safe per backbone.

Both checkpoints scored all four wild-types: `esm2_t12_35M_UR50D` (102 s) and
`esm2_t33_650M_UR50D` (~32 min across four calls). 141,140 of 141,144 rows scored;
the single skipped mutation is the known F64L-parent dissenter from step 6.

### 11a Zero-shot log-odds — a clean NEGATIVE

Four arms, identical folds to steps 5/8/9/10a, all refitted here so every
comparison is exactly paired. Spearman at never-seen positions:

| arm | all novel | single mutants |
|---|---|---|
| dense (66) | 0.4060 | 0.4377 |
| dense+esm 650M (70) | 0.4128 | 0.4440 |
| dense+evo (77) | 0.4506 | 0.4785 |
| dense+evo+esm (81) | 0.4550 | 0.4870 |

Paired bootstrap vs the step-9 winner: `dense+esm − dense+evo = −0.0345,
CI [−0.0511, −0.0172]` (significantly WORSE); `dense+evo+esm − dense+evo = +0.0085,
CI [−0.0028, +0.0199], P(>0) = 0.935` — **not significant**.

Scaling 35M → 650M bought nothing (the 35M's +0.0140 was, if anything, the better
of the two). **Step 9's hypothesis — millions of sequences should beat three
homologs — is refuted for the zero-shot formulation.**

### 11b ESM-2 EMBEDDINGS — the result that changes the project

The open item asked for *embeddings*, which are a different signal: a log-odds
scalar says how likely a substitution is, an embedding says what kind of site it
is. Embedding every variant needs 141k forward passes and is unaffordable, so the
same trick as everywhere else in this project: embed only the four WILD-TYPES
(4 forward passes), take per-residue hidden states, PCA-reduce (fitted on the
wild-type residues only — no labels, fold-safe), and let a variant inherit the
embeddings of the positions it mutates, aggregated mean + spread. Constant per
(backbone, position) — exactly like the evo block.

| arm | all novel | single mutants |
|---|---|---|
| dense+evo (77) | 0.4506 | 0.4785 |
| dense+evo+esm (81) | 0.4550 | 0.4870 |
| **dense+evo+emb (109)** | **0.4906** | **0.5185** |
| dense+evo+esm+emb (141) | 0.4933 | 0.5153 |

`dense+evo+emb − dense+evo = +0.0401, CI [+0.0249, +0.0554], P(>0) = 1.000` —
**significant**, and comparable to the evo block's own +0.044. Adding the zero-shot
scores on top makes it slightly worse, so **embeddings subsume the log-odds signal**.

### 11c Width sweep — negative, 16 dims stands

PCA at 8 / 16 / 32 / 64 components (retaining 34-76% of variance): rho_single
0.5197 / 0.5185 / 0.5204 / 0.5188. Every paired bootstrap against 16 dims spans
zero. **Performance does not depend on the width.** Consistent with finding 8 —
this project's tuning knobs keep coming back flat.

### 11d Leave-one-backbone-out — and the ppluGFP2 question answered

Step 9d's finding was that ppluGFP2 (18-25% identity, the phylogenetic outlier)
is the one backbone the evo features actively hurt, and predicted a language
model would not have that blind spot.

| features | amacGFP | avGFP | cgreGFP | ppluGFP2 | mean |
|---|---|---|---|---|---|
| dense (66) | 0.421 | 0.623 | 0.539 | 0.339 | 0.481 |
| dense+evo (77) | 0.486 | 0.666 | 0.613 | 0.314 | 0.520 |
| dense+evo+esm (81) | 0.465 | 0.672 | 0.604 | 0.323 | 0.516 |
| **dense+evo+emb (109)** | **0.487** | **0.675** | **0.657** | **0.393** | **0.553** |

**ppluGFP2 — the phylogenetic outlier the evo features actively HURT (0.339 → 0.314)
— goes to 0.393 with embeddings**, beating plain dense for the first time in the
project. Step 9 predicted a language model would not have that blind spot. It was
right; it was just wrong about which ESM formulation would deliver it.

### 11e The headline model changes

**`dense+evo+emb` (109 features) replaces `dense+evo` as the recommended feature set
for any never-assayed-position or new-protein task.** Both extrapolation benchmarks
(position CV and LOBO) confirm the same ordering with bootstrapped significance.

Finding 5 (regime-dependent features) and finding 6 (the extrapolation number is
dead/alive separation) both still hold — this changes which features win *within*
extrapolation, not what the ~0.5 ceiling means.

---

## Step 12 — re-optimising and stress-testing the design oracle

### 12a Hyperparameter re-tune — negative, correctly

Step 10's design oracle inherited step 8's configuration, which was selected for a
different problem (extrapolation, dense features only). Recombination is
interpolation over a 6,636-feature block and had never been tuned.

Protocol, per the standing convention: a 10-config grid × 2 feature sets
(with/without the ESM embedding block) scored on the random split's **validation**
rows (13,589), one winner confirmed on **test** (13,581), selection objective =
mean measured brightness of the top 20 (continuous; precision@20 at a 0.5% base
rate is far too noisy to select on).

Validation picked `dense+evo+sparse+emb` with shallower trees (depth 6, 400 trees).
Confirmed on test against the step-10 incumbent:

| metric | tuned − incumbent | 95% CI | P(>0) |
|---|---|---|---|
| mean y@20 | +0.0604 | [−0.0741, +0.2230] | 0.785 |
| precision@20 at y≥1.5 | +0.0286 | [−0.1500, +0.2000] | 0.521 |

**Fails the project's own bar (finding 8). The step-10 oracle configuration is kept
unchanged and the step-10 panel stands.** Step 8's "tuning is exhausted" now holds in
the recombination regime too — which it had not been tested in. Note also that the
ESM embedding block, decisive for extrapolation, does NOT significantly help the
design regime: the sparse block is already doing the site-specific work there.

### 12b A harder, group-disjoint stress test

The design-regime benchmark (step 10c) evaluates on a RANDOM held-out set of
combinations, which is the right regime but the easy end of this project's
evaluations — a test combo at positions {A,B,C} likely has close relatives
{A,B}, {A,C}, {B,C} in training. This groups rows by (backbone, exact set of
mutated positions) and splits whole GROUPS 80/20, so no position-set straddles
train/test. A separate, independently-seeded split (SEED=11) — not paired
with the main benchmark, read as a magnitude check.

Precision@20 under the group-disjoint split (27,136 evaluation rows):

| threshold | dense+evo | +sparse | +sparse+emb |
|---|---|---|---|
| y >= 1.0 | 1.00 | 0.95 | 0.95 |
| y >= 1.2 | 1.00 | 0.95 | 0.95 |
| **y >= 1.5** | **0.40** | **0.85** | **0.80** |

The sparse-block finding **survives the harder split** — if anything the
absolute precision is higher than the original random-split benchmark's 0.45,
because this split's larger test set still allows subset/superset
generalisation (train has {A,B}, test has {A,B,C}), which is the intended
mechanism of recombination design, not a leak. The step-10 panel's expected
yield is not weakened by this check.

### 12c Two small open items, closed

**Log-transform the target — tested, NEGATIVE.** Note the trap, because this item
was briefly closed the wrong way. It is true that Spearman(y, log1p(y)) = 1 exactly,
since Spearman is invariant under any monotonic transform — but that does NOT make
the experiment unnecessary. Training on a transformed target changes the loss, hence
the fitted trees, hence the predictions, so the rank correlation of the PREDICTIONS
against truth can and does move. Measured on the random split with dense+evo+emb:

| target | rho | R2 | RMSE | rho among functional |
|---|---|---|---|---|
| **raw y** | **0.8232** | **0.6802** | **0.2367** | **0.4673** |
| log1p(y) | 0.8181 | 0.6734 | 0.2392 | 0.4369 |

Worse on every metric, and clearly worse among functional variants. **Keep the raw
target.** (The invariance argument would have been a valid reason to skip the test
only if the transform were applied to the predictions rather than to the training
target — it is not the same experiment.)

**Mammalian dataset — excluded.** `mammalian/GFP_variants_mammalian_only.xlsx` is 13
rows of hand-picked named variants (EGFP, sfGFP, mGreenLantern, mNeonGreen,
Clover...) pulled from different papers, each on a different assay (flow
cytometry a.u., % positive cells, fold-vs-EGFP by microscopy) in different
cell types — 8 distinct engineered FPs across 7 assay methods, 10 usable labels.
Both too small to train on and exactly the cross-assay incomparability problem
finding 1 established, and mNeonGreen is not even an avGFP-lineage protein. It is
not a DMS library of the four backbones, so it cannot enter the modelling pipeline
at all. **Recommendation: do not merge into training.**

It is useful instead as an independent external check — diffing its sfGFP sequence
against avGFP WT reproduces the same nine substitutions used as the step-10e
literature positive control (S30R, Y39N, S65T, F99S, N105T, Y145F, M153T, V163A,
I171V, A206V), confirming that control from a source that never touched the DMS
assay. V163A and Y39N are still rank 1-2 of 1,778 in the oracle's ranking.

Files for steps 11-12: `23_esm_scores.py` (patched with the HF backend),
`24_esm_embed.py`, `25_esm_dims_sweep.py`, `26_esm_benchmark.py`, `27_lobo_esm.py`,
`28_design_optimize.py`, `30_make_esm_report.py`. Outputs: `esm.png`,
`esm_benchmark_35M.csv`, `esm_benchmark_650M.csv`, `esm_bootstrap_*.csv`,
`esm_dims_sweep.csv`, `lobo_esm.csv`, `design_opt_grid.csv`,
`design_opt_confirm.csv`.

---

## Step 13 — how much of the design result is near-neighbours?

The step-10 benchmark is the correct regime for recombination but the optimistic end
of this project's evaluations: a held-out 3-mutant may sit one substitution away from
a training variant. Splitting the 13,581 design-regime test rows by whether any
training variant of the same backbone shares all but one of their substitutions:

| precision@20 | all (13,581) | has neighbour (7,060, 52%) | no neighbour (6,521, 48%) |
|---|---|---|---|
| y ≥ 1.0× | 0.95 | 1.00 | 0.80 |
| y ≥ 1.2× | 0.80 | 0.90 | 0.65 |
| y ≥ 1.5× | 0.45 | 0.35 | 0.40 |

Paired bootstrap of the gap: at y≥1.2 `−0.2150, CI [−0.5000, +0.0500]`; at y≥1.5
`+0.0397, CI [−0.3000, +0.3500]`. Both CIs span zero at k=20.

**Read: the honest range for a 20-variant plate is 0.65–0.90 at ≥1.2× WT, and ~0.35–0.45
at ≥1.5×, with the lower end applying to designs with no close relative in the library.**
Even the pessimistic end is ~21× over the 3.0% base rate. The headline claim survives;
quote the range, not the single number.

Files: `29_design_stress_test.py`. Outputs: `design_stress_test.csv`,
`design_stress_bootstrap.csv`.

---

## Open items closed in steps 11-13

| item | verdict |
|---|---|
| Run ESM-2 | **Done.** Zero-shot: negative. Embeddings: +0.040 rho, significant — new headline model |
| Two-stage classifier + regressor (step 10a) | Done — marginal, do not build on it |
| Re-tune the design oracle | Done — **negative**, panel unchanged |
| Log-transform the target | **Negative.** rho 0.8181 vs 0.8232 raw; worse among functional (0.4369 vs 0.4673) |
| Mammalian dataset | **Excluded.** 13 rows, 8 distinct engineered FPs (incl. mNeonGreen, a different lineage), 7 assay methods, 10 usable labels. Not DMS variants of the four backbones; cannot enter training |
| Near-neighbour stress test | Done — honest range recorded above |

## Project status: both halves of the objective are now answered and re-verified.

Remaining open items are all deprioritised by choice, not by blocker: a real MSA of
the GFP family (the one remaining idea with headroom), and a full-scale MLP tune,
which only affects the interpolation ceiling (already rho 0.911, and not what the
design deliverable uses). FPbase harvesting is optional enrichment. Nothing is
blocked on the environment any longer.
