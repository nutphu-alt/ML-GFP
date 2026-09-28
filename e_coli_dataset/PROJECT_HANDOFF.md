# GFP brightness prediction — project handoff

**Purpose of this file.** A complete, self-contained briefing for someone (or some session) picking this project up cold. It covers what the project is for, what has been done, what was found, what the code does, what the environment will and will not allow, and what to do next. It duplicates some of `EDA_summary.md` deliberately — that file is a chronological working log written step by step, whereas this one is organised for a reader who has never seen the project.

**Last updated:** 28 September 2026. Steps 1-13 complete; 28 Sep was an
editorial pass only (findings renumbered 1-8 in reading order, and the duplicated
steps 11-12 sections in `EDA_summary.md` reconciled). No numbers changed.
**Status:** Thirteen steps complete. Both halves of the objective are answered and re-verified. ESM-2 (previously blocked) has run: its zero-shot scores were a clean negative, but its per-residue embeddings are now the best feature set for any never-assayed-position or new-protein task, replacing the step-9 evo-only model. The design oracle was re-tuned (no gain, kept as-is) and stress-tested on a harder split (the sparse-block finding survives). Nothing remains blocked by the environment.

---

## 1. What this project is

The dataset is a set of **deep mutational scanning (DMS)** experiments on green fluorescent protein, published by Sarkisyan et al. (2016) and Gonzalez Somermeyer et al. (2022). Researchers took four different naturally occurring GFPs, generated large libraries of random mutants of each, expressed them in *E. coli*, and measured how brightly each variant fluoresced using FACS-seq (fluorescence-activated cell sorting followed by sequencing).

The result is 141,150 protein variants, each with an amino-acid sequence, a list of its mutations relative to its parent, and a measured brightness.

**The goal is to predict brightness from sequence.** If a model can do that reliably, candidate variants can be ranked computationally and only the promising ones taken to the bench — which is the expensive step. This is the standard motivation for protein fitness prediction, and GFP is the classic benchmark system for it because brightness is easy to measure at scale.

### The four backbones

| Backbone | Organism | WT length | DMS rows |
|---|---|---|---|
| avGFP | *Aequorea victoria* (jellyfish) | 238 aa | 51,715 |
| amacGFP | *Aequorea macrodactyla* | 238 aa | 33,511 |
| ppluGFP2 | *Pontellina plumata* (copepod) | 222 aa | 31,402 |
| cgreGFP | *Clytia gregaria* | 235 aa | 24,516 |

These are homologs, not variants of one protein — pairwise identity ranges from 83% (amacGFP–avGFP) down to 18% (cgreGFP–ppluGFP2). That spread turns out to matter a great deal (see step 9).

Most variants carry 2–3 mutations (range 0 to 43), so the libraries are low-order combinatorial mutants rather than deeply mutated sequences.

---

## 2. The objective, and how success is measured

**Objective:** given an amino-acid sequence (or equivalently a backbone plus a list of mutations), predict fold-wild-type brightness.

There are really *two* objectives, and the project's central result is that they are different problems with different best models:

1. **Interpolation** — ranking new combinations of mutations that have already been individually characterised. Useful for prioritising within a library you have already partly measured.
2. **Extrapolation** — scoring mutations at positions that were never assayed, or in a protein that was never assayed. This is the harder and far more valuable problem, because it is what "design something genuinely new" requires.

**Primary metric is Spearman ρ** (rank correlation), not RMSE or R². This is a deliberate choice forced by a data property described in section 5: brightness values are not comparable across the four libraries even after normalisation, so any metric sensitive to absolute scale is misleading when pooling backbones. RMSE and R² are still reported *within* a single backbone/split where they are meaningful.

---

## 3. The plan, and where we are

| Step | What | Status |
|---|---|---|
| 1 | Merge, clean, tag the 15 source files | Done |
| 2 | Design and verify train/val/test splits | Done |
| 3 | Feature engineering + baseline models | Done |
| 4 | Eight-model comparison with hyperparameter search | Done |
| 4d | Does the winner survive real extrapolation? | Done |
| 5 | Cross-validated extrapolation benchmark (settles 4d) | Done |
| 6 | ESM-2 zero-shot scoring pipeline | **Written, blocked on weights** |
| 7 | Leave-one-backbone-out (cross-protein transfer) | Done |
| 8 | Retune XGBoost for extrapolation | Done — negative result |
| 9 | Cross-homolog evolutionary features | Done — **best result of the project** |
| 10 | Design / optimisation — the second half of the objective | Done — **deliverable produced** |
| 10a | Two-stage classifier + regressor (was open item #3) | Done — answered, marginal |
| 11 | ESM-2, unblocked — zero-shot scores AND embeddings | Done — **embeddings are the new best model** |
| 12 | Re-tune + stress-test the design oracle; close 2 small items | Done — kept as-is, confirmed |
| 13+ | See section 9, "What to do next" | Open |

---

## 4. What was done and what came out of it

### Step 1 — cleaning (`01_clean_gfp_data.py`)

Merged 15 `.xlsx` files into 141,469 rows, then dropped 3 sequences with unresolved `X` residues and 316 rows with no usable brightness label. The 316 comprise 49 standalone natural wild-types (no same-family WT to normalise against) and 264 ML-designed variants measured by colony microscopy with no published WT reference. **Final: 141,150 rows** — 141,144 DMS plus 6 engineered classics (EGFP, sfGFP, Emerald, mEGFP, GFP S65T, avGFP WT reference) held separately because they were measured by molecular brightness (EC×QY) and sit on a completely different scale (19.8–54.1 vs 0–2.5).

A tagging bug was found and fixed here: grouping rows by their *classification label* mis-filed four legitimate FACS-seq wild-type rows into the engineered group. The fix was to group on the **analysing method** instead, since brightness is only comparable within one assay. Worth remembering as a pattern — the physically meaningful field beat the descriptive one.

### Step 2 — splits (`02_make_splits.py`)

Three schemes, all excluding the 6 engineered rows:

- **`split_random`** — stratified by backbone × dark/bright, duplicate-aware. 112,914 / 14,115 / 14,115. The optimistic upper bound.
- **`split_position_holdout`** — entire mutated positions withheld per backbone, so training never sees a mutation at a held-out position. 83,986 / 27,609 / 29,549. The honest extrapolation test.
- **`split_backbone_holdout_eligible`** — flags rows for leave-one-backbone-out.

Why this matters enormously here: DMS variants differ from their parent by 2–3 mutations, so near-identical sequences are everywhere. A plain random split scatters them across train and test, and a model can score well purely by memorising. The gap between the random and position-holdout numbers *is* the measure of how much a model is memorising.

Verified: zero training rows touch a held-out position in any backbone; zero duplicate-sequence leakage in the random split. Two apparent "leaks" in position-holdout are a genuine biological curiosity rather than a bug — an avGFP wild-type sequence is byte-identical to a heavily humanised 38-mutation amacGFP variant.

### Step 3 — features and baselines (`03_build_features.py`, `04_train_baselines.py`, `05_analyze_results.py`)

Two feature blocks were built, and the distinction between them runs through the whole project:

- **Sparse (6,559 columns)** — one-hot `(backbone, position, mutant AA)` indicators. Powerful within a known library, **structurally useless for transfer**: for an unseen position or protein, every one of its columns is zero throughout training.
- **Dense (66 columns)** — transferable descriptors: BLOSUM62 substitution scores, Kyte-Doolittle hydropathy, residue volume, charge, polarity deltas (aggregated sum/mean/min/max), amino-acid composition, mutation count, backbone one-hot, position-fraction statistics.

Baseline results (Ridge, best alpha on validation):

| | random split | position holdout |
|---|---|---|
| mean-by-backbone floor | ρ 0.303 | ρ 0.366 |
| ridge, dense only | ρ 0.610 | ρ 0.666 |
| ridge, sparse only | ρ 0.878 | ρ 0.792 |
| **ridge, combined** | **ρ 0.880 / R² 0.705** | **ρ 0.814 / R² 0.552** |

Two failure modes were diagnosed: truly dead variants get predicted at 0.5–0.75 rather than 0 (an additive model cannot express "one lethal mutation kills the protein"), and accuracy falls steadily as more of a variant is novel — on pure extrapolation, sparse-feature ridge scores *exactly the mean-predictor floor*, because a never-seen position has no fitted coefficient.

### Step 4 — eight model families, 104 configurations (`06_compare_models.py`, `07_summarize_comparison.py`, `08_train_mlp_full.py`)

Ridge, Lasso, ElasticNet, LinearSVR, kernel SVR, RandomForest, XGBoost and an MLP, each hyperparameter-searched, selected on validation only. Run in two tiers because RandomForest and kernel SVR cannot train on 113k rows: a `full` tier and a `sub12k` tier that trains on 12,000 rows but scores the same full test set.

Full-tier test results:

| model | random ρ | random R² | position ρ | position R² |
|---|---|---|---|---|
| **MLP (512,256)** | **0.911** | **0.870** | **0.823** | **0.621** |
| LinearSVR | 0.882 | 0.692 | 0.800 | 0.403 |
| Ridge | 0.880 | 0.705 | 0.814 | 0.552 |
| Lasso | 0.854 | 0.653 | 0.793 | 0.421 |
| ElasticNet | 0.854 | 0.653 | 0.793 | 0.421 |
| XGBoost | 0.813 | 0.665 | 0.786 | 0.609 |

RandomForest (ρ≈0.68) and kernel SVR (ρ≈0.66) were weakest, both struggling with high-dimensional sparse binary input. L1 actively hurts — both Lasso and ElasticNet chose the smallest penalty offered.

**The MLP is the best overall model**, and its R² advantage (0.870 vs ridge's 0.705) is much wider than its ranking advantage, which says it is capturing the non-additive dark mode that defeated the additive models rather than merely re-ranking.

### Steps 4d and 5 — the extrapolation question (`09_analyze_extrapolation.py`, `10_cross_val_extrapolation.py`, `11_analyze_extrapolation_cv.py`)

Breaking the position-holdout test set down by how *novel* each row is reversed the ranking. Step 4d found XGBoost best on the 293 pure-extrapolation single mutants, but with a bootstrap CI that crossed zero — suggestive, not established.

Step 5 settled it properly. The dataset contains 4,596 single mutants, not 293; the benchmark was small only because the holdout withheld 6% of positions. **Five-fold cross-validation over positions** (each fold trains on rows whose mutations all sit outside the fold, tests on rows whose mutations all sit inside it, boundary rows dropped from both) tests every variant exactly once by a model that never saw any of its positions — 15.7× more evidence from the same data.

Out-of-fold Spearman at never-seen positions:

| subset | n | ridge | **xgboost** | mlp |
|---|---|---|---|---|
| all-novel variants | 16,017 | 0.367 | **0.422** | 0.339 |
| **single mutants** | **4,596** | 0.368 | **0.441** | 0.329 |
| single — amacGFP | 1,201 | 0.251 | **0.384** | 0.177 |
| single — avGFP | 1,085 | 0.353 | **0.442** | 0.297 |
| single — cgreGFP | 1,169 | 0.395 | **0.496** | 0.425 |
| single — ppluGFP2 | 1,141 | 0.299 | **0.301** | 0.284 |

Paired bootstrap (4,000 resamples): xgboost − mlp +0.112 [+0.091, +0.133]; xgboost − ridge +0.072 [+0.054, +0.091]; ridge − mlp +0.039 [+0.017, +0.061]. **All three significant.**

So **XGBoost is decisively the best extrapolator, and the MLP — the best headline model — is significantly *worse than plain ridge* at unseen positions.** The reading is that the MLP's extra capacity goes into learning the specific mutation landscape it was shown, which is exactly what fails to transfer.

### Step 6 — ESM-2 pipeline (`23_esm_scores.py`, `12_derive_wt_sequences.py`)

Written and dry-run validated but **never executed against real weights** — see section 7 for why. It implements the standard zero-shot formulation (Meier et al. 2021): `score = log P(mut at p) − log P(wt at p)`, needing only ~L forward passes *per backbone*, not per variant.

A useful side result: each backbone's wild-type sequence was reconstructed by independently reverting every variant's own mutations and taking a consensus. **141,141 of 141,142 reversions agree.** The single dissenter is the `avGFP (parent, F64L)` row, which confirms that this dataset's mutation numbering is relative to the **F64L library parent**, not literal wild-type avGFP. Stored in `output/backbone_wt.json`.

### Step 7 — leave-one-backbone-out (`13_lobo_eval.py`)

Train on three backbones, predict the fourth — the hardest transfer test.

| features | model | amacGFP | avGFP | cgreGFP | ppluGFP2 | mean |
|---|---|---|---|---|---|---|
| dense (66) | ridge | 0.308 | 0.609 | 0.544 | 0.354 | 0.454 |
| dense (66) | **xgboost** | 0.421 | 0.623 | 0.539 | 0.339 | **0.481** |
| dense (66) | mlp | 0.318 | 0.559 | 0.552 | 0.352 | 0.445 |
| combined (6,625) | ridge | 0.293 | 0.624 | 0.544 | 0.336 | 0.449 |
| combined (6,625) | **xgboost** | 0.365 | 0.635 | 0.566 | 0.365 | **0.483** |

Third independent confirmation that XGBoost generalises best. Also: **the 6,559 sparse indicators contribute essentially nothing** (0.481 dense vs 0.483 combined) — exactly as their construction predicts. And cross-protein transfer (ρ≈0.48) is about as hard as predicting an unseen site in a known protein (ρ≈0.44), not dramatically harder.

### Step 8 — retuning XGBoost (`14_tune_xgb_extrapolation.py`) — a negative result

Every XGBoost number until this point used a config carried over from the step-4 grid, which selected on the random split — the regime XGBoost is worst at. A 48-config grid was screened on 2 of the 5 folds and looked promising (shallower trees, ρ 0.461 vs ~0.42). **Checked properly across all 5 folds, the gain vanished:** best candidate 0.4393 vs default 0.4377, Δ=+0.0015, 95% CI [−0.015, +0.018].

The phase-1 "win" was 2-fold sampling noise. **The untuned default is already as good as anything in this hyperparameter neighbourhood for extrapolation.** This is the same lesson as step 4d → step 5, learned again: small-sample extrapolation checks overstate differences.

A confirmation fell out of it — dense-only XGBoost scored 0.4377 against step 5's combined-feature 0.441 on identical folds, so the sparse block can be dropped for novel-position work too, not just cross-protein work, at 2–4× the speed.

### Step 9 — evolutionary information breaks the plateau (`16_align_backbones.py`, `17_evo_features.py`, `18_evo_benchmark.py`, `19_evo_lobo.py`)

**This is the most important result in the project.**

Everything had plateaued near ρ≈0.44 at never-assayed positions for one reason: nothing in the 66 descriptors describes a *specific site*. BLOSUM62 knows how often Leu replaces Ile across all proteins; it does not know that position 66 is the chromophore tyrosine and must never change. Step 8 closed off hyperparameter tuning as a route. The only remaining route was to import site-specific knowledge from outside the assay.

ESM-2 was the intended source and is unreachable offline. So step 9 used the outside information **already sitting in the dataset**: the four backbones are homologous GFPs from four organisms, so their differences are an evolutionary record of what a GFP tolerates at each site.

**9a — alignment.** Needleman-Wunsch with BLOSUM62 and affine gaps (open −11, extend −1), star topology against avGFP. 255 columns, every residue of every backbone mapped. Two independent validations passed: the chromophore Tyr66-Gly67 lands in the same two columns in all four backbones, and pairwise identities reproduce the known phylogeny (amacGFP–avGFP 82.8%, cgreGFP ~44–45% to both, ppluGFP2 the outlier at 17.8–24.9%).

**9b — eleven features.** For a mutation wt→mut at position *p* on backbone *B*, with *H* = the residues of the other three backbones at the aligned column: conservation (fraction of *H* equal to wt), `mut_in_homolog` (has nature already made this exact substitution?), and `evo_score` = mean BLOSUM62(mut, *H*) − mean BLOSUM62(wt, *H*) — deliberately the same log-odds shape as the ESM score. Aggregated per variant. **No label information enters these**, so they cannot leak across a fold.

**9c — the benchmark.** Identical folds to step 5, identical XGBoost config, identical metric. Baseline predictions reused from step 8 so the comparison is exactly paired.

| subset | n | dense (66) | dense+evo (77) | Δ |
|---|---|---|---|---|
| all-novel variants | 16,017 | 0.4060 | **0.4554** | **+0.049** |
| **single mutants** | **4,596** | 0.4377 | **0.4822** | **+0.044** |
| single — amacGFP | 1,201 | 0.4058 | 0.4330 | +0.027 |
| single — avGFP | 1,085 | 0.4874 | 0.5391 | +0.052 |
| single — cgreGFP | 1,169 | 0.4763 | **0.6251** | **+0.149** |
| single — ppluGFP2 | 1,141 | 0.2879 | 0.2567 | −0.031 |

Paired bootstrap, single mutants: **Δρ = +0.0444, 95% CI [+0.0275, +0.0603], P(Δ>0) = 1.000 — significant.** `evo_score_sum` became the single highest-gain feature in the whole model, and the evo block carries 17.7% of total importance on 14.3% of the width.

**9d — on a new protein.** Leave-one-backbone-out, same protocol as step 7:

| features | amacGFP | avGFP | cgreGFP | ppluGFP2 | mean |
|---|---|---|---|---|---|
| dense (66) | 0.421 | 0.623 | 0.539 | 0.339 | 0.481 |
| dense + evo (77) | **0.485** | **0.675** | **0.618** | 0.317 | **0.524** |
| Δ | +0.064 | +0.052 | +0.079 | −0.022 | **+0.043** |

**ppluGFP2 is the informative exception.** It is the only backbone that degrades, and it does so in *both* experiments. It is also the phylogenetic outlier at 18–25% identity — at that distance the aligned columns carry little real constraint, so the features are noise for it. cgreGFP gains most, and it is the one with two relatives at ~45% identity: close enough to align confidently, distant enough that a shared residue means evolution actually held the site fixed. **The benefit tracks having relatives at intermediate evolutionary distance.**

---

### Steps 10-12 — design, ESM-2 unblocked, re-optimisation (`21_design_variants.py`, `23_esm_scores.py`, `24_esm_embed.py`, `26_esm_benchmark.py`, `27_lobo_esm.py`, `28_design_optimize.py`, `29_design_stress_test.py`)

Full detail in `EDA_summary.md`; this is the compressed version.

**Step 10 — design by recombination.** Maximising the step-9 model over
candidate sequences was checked before being trusted, and failed: at
never-assayed positions the ~0.45 ρ is almost entirely dead/alive separation
(functional-only ρ 0.247, above 1.0× WT it is -0.146, top-20 enrichment for
bright variants < 1). Not fixable by a two-stage classifier+regressor (step
10a, marginal gains only). Not a data ceiling either — on the random split,
precision@20 for ≥1.2× WT is 1.00 against a 3.0% base rate — so design moved
to recombining already-characterised substitutions, the regime where the
model works. That reversed which features matter: the sparse
(backbone,position,mutant) block, retired as dead weight for extrapolation in
steps 4/7/8, is what finds the brightest recombinants (+0.372 precision@20 at
y≥1.5, bootstrapped). Stacking only destroys brightness when components are
unscreened; restricting the extension pool to substitutions individually
measured ≥0.95× WT fixes it. The resulting panel (`design_panel.csv`, 23 avGFP
variants) independently rediscovers the superfolder GFP mutation set with no
literature input.

**Step 11 — ESM-2 unblocked.** `huggingface.co` was reachable from a later
cloud session even though the original sandbox's blocked hosts
(`dl.fbaipublicfiles.com`, `download.pytorch.org`) stayed blocked. Zero-shot
masked-marginals scores (the formulation step 6 specified) are a clean
negative — worse than the evo features alone, nothing added on top. Per-residue
*embeddings* from the same weights are not: +0.040 ρ on unseen positions
(bootstrapped, CI [+0.025,+0.055]) and +0.079 mean ρ on leave-one-backbone-out,
finally fixing the ppluGFP2 blind spot step 9 predicted a language model
would fix (0.314 → 0.393). `dense+evo+emb` (109 features) is now the
recommended set for any extrapolation task, replacing step 9's `dense+evo`.

**Step 12 — re-optimising and stress-testing.** The design oracle's XGBoost
config (inherited from step 8, tuned for a different regime) was re-tuned
against a val-selected/test-confirmed grid: no significant gain (CI spans
zero), so it was kept unchanged. The design-regime benchmark was re-run on a
harder split that separates the 13,581 design-regime test rows by whether any
training variant of the same backbone shares all but one of their
substitutions — i.e. whether a near neighbour exists to copy from. 52% have
one, 48% do not. precision@20 falls from 0.90 to 0.65 at y≥1.2 without a
neighbour, and holds at ~0.35-0.40 at y≥1.5; both bootstrapped gaps span zero
at k=20. Two small open items were closed: log-transforming
the regression TARGET was tested and does not help (see item 5 in section 9),
and the mammalian dataset (13 rows, heterogeneous assays) was decided against
merging into training, per finding 1.

## 5. Eight findings that constrain everything else

These are the load-bearing conclusions. A new session should read these before designing any new experiment.

**1. Brightness is not comparable across libraries, even after normalisation.** Two byte-identical protein sequences appear in two different DMS libraries (an amacGFP variant with 41 mutations converged to exactly the avGFP F64L sequence). The same protein reads raw MFI 5,238 in one and 2,292 in the other; after fold-WT normalisation it reads **1.000 versus 0.245** — normalising made the gap *worse*, 2.3× raw becoming 4.1×. Worse, the same point mutation (H169L) on that identical background moves brightness in **opposite directions** in the two libraries (0.865× vs 1.540×). Since the background sequence is literally identical, this is assay disagreement rather than biology, and it puts a floor on achievable accuracy. **Consequence: score anything cross-backbone with Spearman only. Never RMSE or R² across libraries.**

**2. The label is strongly bimodal.** About 34% of variants are non-functional — a spike near 0 and a peak near 1.0, in all four backbones. An additive model cannot express "one lethal mutation kills the protein", which is precisely where ridge fails and where the MLP's R² advantage comes from.

**3. The best model depends on the task, and the two answers are opposite.** The MLP wins interpolation (ρ 0.911 random) and *loses to plain ridge* on extrapolation (0.329 vs 0.368). XGBoost loses the headline comparison (0.813, last of the linear-and-up models) and wins extrapolation decisively (0.441). Any statement of the form "model X is best here" must name the regime.

**4. The sparse feature block is dead weight for any transfer task.** It is 6,559 of 6,625 columns and contributes nothing once positions or proteins are unseen (confirmed in step 7 cross-protein and step 8 within-protein). Dropping it costs nothing measurable and is 2–4× faster.

**5. Which features help is REGIME-dependent, and the answer reverses.** Finding 4 above (the sparse block is dead weight) is correct for extrapolation and wrong for recombination. Designing by recombining already-characterised substitutions is the interpolation regime, and there the sparse block is what finds the brightest variants: precision@20 at y ≥ 1.5 goes 0.15 → 0.45 when it is added (paired bootstrap +0.372, 95% CI [+0.100, +0.650]). Step 10's oracle is the only model in this project that uses it. State the regime before stating which features matter.

**6. The ~0.45 extrapolation Spearman is dead-vs-alive separation, not brightness ranking.** Restricted to functional variants it is 0.247, and above 1.0× WT it is *negative*. Any downstream use that ranks among functional variants at unseen positions is using a number that does not mean what it appears to mean. See step 10.0.

**7. Zero-shot log-odds and embeddings are not the same signal, and only one paid off.** ESM-2 650M masked-marginals scores are significantly worse than the evo features alone (-0.0345 ρ) and add nothing on top of them. Per-residue embeddings from the identical weights add +0.040 ρ (CI [+0.025,+0.055]) and fix the one backbone (ppluGFP2) the evo features hurt. A "does ESM-2 help" question has two different answers depending on which output of the model you use — check both before concluding either way.

**8. Small-sample extrapolation checks overstate differences — twice burned.** Step 4d's 293-row result had a CI crossing zero and was called "suggestive"; step 5's 4,596-row version confirmed and strengthened it. Step 8's 2-fold screen produced an apparent +0.04 win that evaporated entirely on 5 folds. **Always run the full 5-fold pooled benchmark with a paired bootstrap before believing an extrapolation result.**

---

## 6. Code inventory

All scripts live in `pipeline/` (the 8 that produce the deliverable) and `experiments/` (the other 22), and resolve paths relative to themselves (`BASE_DIR = Path(__file__).resolve().parent.parent`), so they can be run from anywhere. **Filenames carry a two-digit run-order prefix (`01_` … `31_`), so the folder sorts in execution order** and the table below is that order.

**How the numbered modules import each other.** A Python module name cannot begin with a digit, so the six scripts that are imported by others are loaded by name instead of with a plain `import`:

```python
import importlib, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
ef = importlib.import_module("17_evo_features")
```

The six importable modules and who loads them: `03_build_features` and `17_evo_features` ← `21_design_variants`; `16_align_backbones` ← `17_evo_features`; `18_evo_benchmark` ← `20_two_stage_benchmark`, `25_esm_dims_sweep`, `26_esm_benchmark`, `27_lobo_esm`; `24_esm_embed` ← `25_esm_dims_sweep`; `21_design_variants` ← `28_design_optimize`, `29_design_stress_test`. Only `28` and `29` cross folders (`experiments/` -> `pipeline/`); both insert `pipeline/` on `sys.path` for that reason. **If you renumber a file, update the string in every script that loads it** — the reference is a string, so nothing will flag it until the script runs.

| Script | Step | What it does |
|---|---|---|
| `01_clean_gfp_data.py` | 1 | Merge 15 xlsx → clean → tag backbone/source group → EDA plot |
| `02_make_splits.py` | 2 | Build and verify the three split schemes |
| `03_build_features.py` | 3a | Sparse (6,559) and dense (66) feature matrices |
| `04_train_baselines.py` | 3b | Ridge variants + HistGradientBoosting, both splits |
| `05_analyze_results.py` | 3c | Extrapolation-by-novelty breakdown, baseline figure |
| `06_compare_models.py` | 4a | 104-config, 8-family comparison. Resume-safe |
| `07_summarize_comparison.py` | 4b | Leaderboard (selected on validation) + figure |
| `08_train_mlp_full.py` | 4c | Full-data MLP via manual epoch loop + checkpointing |
| `09_analyze_extrapolation.py` | 4d | Novelty breakdown for the full-data MLP |
| `10_cross_val_extrapolation.py` | 5a | 5-fold position CV. Resume-safe, the core benchmark |
| `11_analyze_extrapolation_cv.py` | 5b | Pools folds, paired bootstrap, figure |
| `12_derive_wt_sequences.py` | 6 | Reconstruct + cross-check the four WT sequences |
| `13_lobo_eval.py` | 7 | Leave-one-backbone-out. Resume-safe |
| `14_tune_xgb_extrapolation.py` | 8 | Grid → eval → analyze phases. Resume-safe |
| `15_plot_xgb_tune.py` | 8 | Figure for step 8 |
| `16_align_backbones.py` | 9a | Needleman-Wunsch star alignment + validation. Exports `BLOSUM62` |
| `17_evo_features.py` | 9b | The 11 cross-homolog features |
| `18_evo_benchmark.py` | 9c | dense vs dense+evo on the step-5 folds + bootstrap + figure |
| `19_evo_lobo.py` | 9d | dense vs dense+evo on leave-one-backbone-out |
| `20_two_stage_benchmark.py` | 10a | Dedicated classifier + functional-only regressor on the step-5 folds |
| `21_design_variants.py` | 10 | Design oracle, design-regime benchmark + bootstrap, beam search |
| `22_make_design_report.py` | 10b | Assembles `design_panel.csv` and `design.png` |
| `23_esm_scores.py` | 11a | Zero-shot masked-marginals; now has `--hf-model` (HuggingFace) alongside `--checkpoint` (fair-esm) |
| `24_esm_embed.py` | 11b | Per-residue ESM-2 embeddings of the 4 wild-types, PCA-reduced, aggregated per variant |
| `25_esm_dims_sweep.py` | 11b | PCA width sweep (flat; 16 kept) |
| `26_esm_benchmark.py` | 11 | All ESM arms on the settled 5-fold CV, paired bootstraps |
| `27_lobo_esm.py` | 11c | Leave-one-backbone-out with ESM arms — the ppluGFP2 result |
| `28_design_optimize.py` | 12 | XGBoost re-tune for the design regime, val-selected/test-confirmed |
| `29_design_stress_test.py` | 13 | Splits the design benchmark by near-neighbour availability |
| `30_make_esm_report.py` | 11-12 | Assembles `esm.png` |
| `31_make_slides.js` | — | Generates the results deck (`pptxgenjs`) |

### Key outputs

`output/gfp_variants_clean.csv` and `gfp_variants_split.csv` are the modelling tables. `output/features/` holds `X_sparse.npz`, `X_dense.npy`, `X_evo.npy`, `y.npy`, `meta.csv` and the feature-name lists — **`meta.csv` row order is the canonical index**; every prediction file aligns to it. `output/backbone_wt.json` and `backbone_alignment.json` hold the sequences and alignment. `output/GFP_ML_results.pptx` is the 17-slide deck.

### Reproducing the fold assignment

Several scripts must use **identical** position folds to stay comparable. The recipe is `assign_position_folds(meta, np.random.default_rng(SEED))` with `SEED = 7`, `N_FOLDS = 5`, operating on `output/features/meta.csv` in its stored order. It is duplicated verbatim in `10_cross_val_extrapolation.py`, `14_tune_xgb_extrapolation.py` and `18_evo_benchmark.py`. **If you refactor it, refactor all three together**, or the cross-step comparisons silently stop being paired.

---

## 7. Environment constraints (read before running anything)

These shaped nearly every implementation decision, and they will bite a new session that does not know about them.

**Wall-clock limit of roughly 180 seconds per shell call.** The tool schema nominally accepts far longer timeouts; the host does not honour them. Anything slower than this must checkpoint to disk and resume across calls.

**Background processes do not survive between calls.** `nohup ... &` is useless — each call gets a fresh process namespace, confirmed by `ps aux` showing no survivors. There is no way to run something long in the background and come back to it.

**The established workaround is resume-safe scripting**, used throughout: append each result to a CSV as it completes and skip already-done work on re-run (`06_compare_models.py`, `13_lobo_eval.py`, `14_tune_xgb_extrapolation.py`), or checkpoint model state with joblib after every epoch (`08_train_mlp_full.py`, `10_cross_val_extrapolation.py`). Scripts carry a `TIME_BUDGET` constant (100–150s) and exit cleanly with a "re-run to continue" message. Expect to invoke some of them a dozen times.

**Network egress is restricted to an allowlist.** `pypi.org` returns 200; `huggingface.co`, `dl.fbaipublicfiles.com` and `download.pytorch.org` all return nothing. This was diagnosed by tracing CONNECT tunnels, and re-verified on 26 Sep 2026 — it is a proxy allowlist, **not** the user's own internet connection. Consequence: `pip install` from PyPI works, but PyTorch's CPU index is blocked and plain `pip install torch` pulls a multi-gigabyte CUDA stack that cannot finish in one call.

**To unblock step 6**, two files need to be placed in the project folder manually (the filesystem mount is reachable even though the network is not): a **Linux** `cp310` PyTorch CPU wheel from `download.pytorch.org/whl/cpu/torch/`, and an ESM-2 checkpoint from `dl.fbaipublicfiles.com/fair-esm/models/` — `esm2_t12_35M_UR50D.pt` (~150 MB) is the recommended first test before `esm2_t33_650M_UR50D.pt` (~2.5 GB). `23_esm_scores.py` then runs with `--checkpoint <path>`; it downloads nothing itself and has been dry-run validated end to end.

**Model-specific compute notes.** HistGradientBoosting cannot accept sparse input at all (it is excluded from the comparison; XGBoost is the gradient-boosting representative). Ridge on sparse input must use `solver="lsqr"`, not `sparse_cg` — 6s versus 18s. Exact Lasso needs ~103s per fit, so Lasso and ElasticNet use `SGDRegressor` with the same penalty. Kernel SVR was capped at `gamma='scale'`, `C ≤ 1` and sub-sampled to 6,000 training rows; its numbers are a lower bound, not a fair comparison. A full-data MLP fit takes ~9 minutes and therefore *must* go through the epoch-checkpointing path.

**Deck regeneration requires a symlink.** From `experiments/`: `ln -sf /sessions/<session>/node_modules node_modules`, then `node 31_make_slides.js`, then remove the symlink. Always validate afterwards with the pptx skill's `validate.py`, and visually check changed slides by converting to PDF and rasterising with `pdftoppm`.

---

## 8. Open items and known issues

**`output/gfp_variants_clean.csv` may be stale.** During the step-1 bug fix it could not be overwritten because an external program (almost certainly Excel, evidenced by a `~$GFP_variants_part02_of_15.xlsx` lock file) held it open. The corrected data was saved alongside. This has not been re-verified since. **Check before trusting that file** — note the downstream pipeline reads `gfp_clean.pkl` and `gfp_variants_split.csv`, which are correct, so no result depends on it.

**475 MB of intermediates sit in `output/`.** `extrap_cv/` (340 MB, per-fold predictions and MLP checkpoints) and `mlp_checkpoints/` (135 MB). The `.npz` prediction files in `extrap_cv/` are still *used* by `18_evo_benchmark.py` for its paired baseline, so do not delete those without regenerating. The `.ckpt` files are safe to remove.

**Step 6 is no longer blocked.** A later cloud session found `huggingface.co` reachable even though `dl.fbaipublicfiles.com` and `download.pytorch.org` stayed blocked, and ran ESM-2 through a HuggingFace backend added to `23_esm_scores.py`. See step 11. Nothing in this project remains blocked by the environment.

**The MLP was never properly tuned at full scale.** Its architecture and alpha were selected on the 12k subsample, and both full-data runs early-stopped at epoch 6 of 10. Its headline numbers are therefore a lower bound — relevant only to the interpolation use case. Left deprioritised: it only affects interpolation, which already sits at ρ 0.911 and is not what the design deliverable uses.

---

## 9. What to do next

After step 12, every item originally listed here is closed. What is left is
genuinely optional — nothing is blocking the project's two stated objectives.

**0. Keep using the panel.** `output/design_panel.csv` is 23 avGFP variants, every
component individually measured, none of them existing library members.
Expected yield is a RANGE, not a point: precision@20 at ≥1.2× WT is 0.65-0.90
depending on whether a design has a near neighbour in the library (step 13),
and ~0.35-0.45 at ≥1.5×. Quote the range. Even the pessimistic end is ~21×
over the 3.0% base rate. Results feed back in as new characterised
substitutions, which widens the recombination pool — the loop that compounds.
The design oracle itself does not need to change: step 12's re-tune gained
nothing (CI spans zero), and the ESM embedding block doesn't move the design
regime either (the sparse block already carries the site-specific signal
recombination needs) — both kept exactly as step 10 built them.

**1. ~~ESM-2 embeddings~~ — DONE in step 11, confirmed on two benchmarks.**
Zero-shot scores were a clean negative (significantly worse than the evo
features alone). Embeddings from the same weights are not: +0.040 ρ on unseen
positions (CI [+0.025,+0.055]), and on leave-one-backbone-out ppluGFP2 — the
one backbone the evo features hurt — finally improves, 0.314 → 0.393. Step 9's
prediction was right about the mechanism (a PLM has no ppluGFP2 blind spot)
and wrong about the formulation (embeddings, not log-odds scores).
`dense+evo+emb` is now the recommended feature set for extrapolation work.

**2. ~~A real MSA of the GFP family~~ — superseded.** This was the fallback for
if ESM stayed blocked. It didn't; ESM-2 embeddings already deliver a larger,
confirmed gain than three-sequence conservation features could plausibly add
on top. Not worth building unless embeddings become unavailable again.

**3. ~~Two-stage model~~ — DONE in step 10a, answered.** A dedicated classifier moves dead/alive AUC 0.778 → 0.791; a regressor fit on functional rows only moves Spearman among functional variants 0.247 → 0.262. Real but marginal, and it does not change the verdict that brightness above functional is unpredictable at unseen positions. Do not build on it.

**4. Tune the MLP properly at full scale** — the one item left genuinely open, and deliberately deprioritised. It would only raise the interpolation ceiling (already ρ 0.911), and interpolation is not what the design deliverable uses. Pick this up only if a future need for it appears.

**5. ~~Log-transform brightness~~ — DONE, tested, NEGATIVE.**
Note the trap: it is true that Spearman(y, log1p(y)) = 1, but that does NOT
make the experiment unnecessary. Training on a transformed target changes the
loss, hence the fitted trees, hence the predictions — so the rank correlation
of the PREDICTIONS against truth can and does move. Measured on the random
split with dense+evo+emb: raw y gives rho 0.8232 / R2 0.6802 / RMSE 0.2367;
log1p(y) gives rho 0.8181 / R2 0.6734 / RMSE 0.2392, and among functional
variants it is clearly worse (0.4369 vs 0.4673). **Keep the raw target.**

**6. ~~Mammalian dataset~~ — DONE in step 13, decided.** 13 rows, heterogeneous
assays across different papers and cell types — too small to train on, and
exactly the cross-assay incomparability problem finding 1 established.
**Recommendation: do not merge it into training.** Its 8 distinct entries are
engineered FPs (EGFP, sfGFP, mNeonGreen, Clover, mGreenLantern) measured by 7
different methods across HEK293/HEK293T/HeLa, and mNeonGreen is not even an
avGFP-lineage protein. It is not a DMS library of the four backbones, so it
cannot enter the modelling pipeline at all. One legitimate future use, NOT yet
performed: scoring its sfGFP sequence as a mutation set against the avGFP
parent would replay the step-10e positive control from a source that never
touched the DMS assay.

**7. Optional: harvest FPbase** (~250 curated green FPs). Its API was
unreachable from the original sandbox; worth a retry now that huggingface.co
turned out to be reachable — untested whether FPbase specifically is.

### Standing conventions to preserve

Score cross-backbone work with **Spearman only**. Select hyperparameters on **validation, never test**. Any new extrapolation claim goes through the **5-fold pooled benchmark with a paired bootstrap** before it is believed. Keep `EDA_summary.md` and the **PowerPoint deck updated after every modelling step** — this was a standing instruction from the user and the deck is the primary deliverable.

### Honest assessment of where this stands

The project now answers both halves of its question, and as of step 12 every
number behind that answer has been re-checked at least once — by a bootstrap,
by a held-out control, or by a harder split.

**Design by recombination works and is validated** — precision@20 of 0.65-0.90
for ≥1.2× WT against a 3.0% base rate, the lower end applying to designs with
no close relative in the library (step 13), and the loop independently
rediscovers the superfolder GFP mutation set with no literature input. **Design by invention
does not** — at never-assayed positions the model ranks foldability, not
brightness, and enrichment for bright variants is below 1. The panel in
`design_panel.csv` lives entirely in the first category. Do not let a future
step quietly migrate it into the second.

**The predictor's headline model changed in step 11, and this is the one
correction a returning reader most needs.** `dense+evo`, step 9's winner, is
no longer the best feature set for never-assayed positions or new proteins —
`dense+evo+emb` (adding ESM-2 embeddings) beats it on both the position-CV
benchmark and leave-one-backbone-out, each confirmed by bootstrap. The deck's
"Where this leaves us" slide and this file's headline tables reflect the
update; anywhere else this project's earlier numbers are quoted from memory,
prefer the ones in section 4/11 of this file.

The project answers its core question. There is a cleaned and verified dataset, three leakage-checked split designs, a comparison across eight model families, a three-times-confirmed answer on which model suits which task, and a significant improvement from evolutionary features.

The one caveat to carry into any downstream use: **even the best extrapolation performance is ρ≈0.48**. That is genuinely useful for triage — it will enrich a shortlist substantially over random picking — but it is not accurate enough to trust an individual prediction. Anyone using this to pick variants for the bench should treat it as a ranking aid, not an oracle.
