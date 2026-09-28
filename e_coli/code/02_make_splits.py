"""
GFP variants dataset: step 2 — build train/val/test splits.

Produces three independent split schemes, each as its own column added to
the cleaned dataset:

  split_random             : stratified random split (baseline / upper bound)
  split_position_holdout   : per-backbone, held-out mutated positions
                              (tests generalization to unseen sites)
  split_backbone_holdout   : just the Backbone column re-used as a fold id
                              (leave-one-backbone-out cross-validation)

Row eligibility:
  - 'engineered/classic' rows (different brightness scale) are excluded from
    every split scheme and labeled "excluded".
  - Everything else is DMS data on a consistent fold-WT brightness scale.

Run with: python make_splits.py   (from anywhere -- paths are resolved
relative to this script's location, not the current working directory)
Input : output/gfp_clean.pkl (written by clean_gfp_data.py)
Output: output/gfp_variants_split.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# this file lives in e_coli/code/, so its parent's parent is the e_coli/ folder
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
IN_PATH = OUT_DIR / "gfp_clean.pkl"
OUT_PATH = OUT_DIR / "gfp_variants_split.csv"

SEED = 42
BRIGHT_THRESHOLD = 0.2          # brightness cutoff separating the "dark" and "bright" modes
RANDOM_VAL_FRAC = 0.10          # fraction of eligible rows held out as validation (random split)
RANDOM_TEST_FRAC = 0.10         # fraction of eligible rows held out as test (random split)
POSITION_VAL_FRAC = 0.06        # fraction of positions per backbone held out as "val positions"
POSITION_TEST_FRAC = 0.06       # fraction of positions per backbone held out as "test positions"
# NOTE: because most variants carry several mutations at once, the probability
# that a row touches *at least one* held-out position rises quickly with
# mutation count (roughly 1-(1-f)^n for n mutations). These fractions were
# chosen empirically so the resulting row-level test/val sizes land near 15%
# each -- see the printed row counts after running.

MUTATION_PATTERN = re.compile(r"[A-Z](\d+)[A-Z]")   # captures the position number inside e.g. "I32V"


def extract_positions(variant_name: str) -> list[int]:
    """Return the list of mutated residue positions mentioned in a variant name."""
    if not isinstance(variant_name, str):
        return []
    return [int(p) for p in MUTATION_PATTERN.findall(variant_name)]


def add_dedup_group(df: pd.DataFrame) -> pd.DataFrame:
    """Assign every row a group id such that identical sequences share the same id.

    This guarantees exact-duplicate sequences never end up split across
    train/val/test (that would leak the answer for one copy into the set
    scoring the other copy).
    """
    df["dedup_group"] = df["Protein sequence"].factorize()[0]
    return df


def add_brightness_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Coarse dark/bright label used only for stratifying splits, not modeling."""
    df["brightness_bin"] = np.where(df["Brightness value"] < BRIGHT_THRESHOLD, "dark", "bright")
    return df


def random_stratified_split(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Group-aware, stratified random split.

    Stratified by (Backbone, brightness_bin) so every backbone and every
    brightness mode is represented proportionally in train/val/test.
    Group-aware by dedup_group so duplicate sequences stay together.
    """
    assign = pd.Series("train", index=df.index, dtype="object")

    # work on one representative row per dedup group, then broadcast the
    # decision to every row sharing that group
    groups = df.groupby("dedup_group").agg(
        Backbone=("Backbone", "first"),
        brightness_bin=("brightness_bin", "first"),
    )

    for _, stratum in groups.groupby(["Backbone", "brightness_bin"]):
        group_ids = stratum.index.to_numpy()
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_test = int(round(n * RANDOM_TEST_FRAC))
        n_val = int(round(n * RANDOM_VAL_FRAC))
        test_ids = set(group_ids[:n_test])
        val_ids = set(group_ids[n_test:n_test + n_val])
        assign.loc[df["dedup_group"].isin(test_ids)] = "test"
        assign.loc[df["dedup_group"].isin(val_ids)] = "val"

    return assign


def position_holdout_split(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Per-backbone split on mutated *positions*, not on rows.

    For each backbone: randomly partition its mutated positions into
    train / val / test position sets. A variant's split is decided by the
    "most held-out" position it touches: if any of its mutations sits at a
    test-position, the whole row goes to test (strictest); else if any
    mutation sits at a val-position, the row goes to val; otherwise train.

    Note this guarantees only that TRAIN rows never touch a held-out
    position -- not that test/val rows are 100% novel. A multi-mutant test
    row can still carry other mutations at ordinary train-positions (that's
    unavoidable with combinatorial variants); what matters is that no
    held-out position is ever *trained on*, which this construction
    enforces exactly.

    Returns (assign, held_out_positions) where held_out_positions records
    the actual test/val position sets per backbone, for verification.
    """
    mutated_positions = df["Variant name"].apply(extract_positions)  # local Series, not stored on df
    assign = pd.Series("train", index=df.index, dtype="object")
    backbones = df["Backbone"]
    held_out_positions: dict = {}

    for backbone in backbones.unique():
        idx = df.index[backbones == backbone]
        positions_here = mutated_positions.loc[idx]
        all_positions = sorted({p for positions in positions_here for p in positions})
        positions_arr = np.array(all_positions)
        rng.shuffle(positions_arr)

        n = len(positions_arr)
        n_test = int(round(n * POSITION_TEST_FRAC))
        n_val = int(round(n * POSITION_VAL_FRAC))
        test_positions = set(positions_arr[:n_test])
        val_positions = set(positions_arr[n_test:n_test + n_val])

        touches_test = positions_here.apply(lambda ps: any(p in test_positions for p in ps))
        touches_val = positions_here.apply(lambda ps: any(p in val_positions for p in ps))

        assign.loc[idx[touches_test.to_numpy()]] = "test"
        assign.loc[idx[(touches_val & ~touches_test).to_numpy()]] = "val"

        # remember exactly which positions were held out, so this can be
        # checked later instead of re-derived (see verify_splits.py)
        held_out_positions[backbone] = {
            "test_positions": sorted(int(p) for p in test_positions),
            "val_positions": sorted(int(p) for p in val_positions),
        }

    return assign, held_out_positions


def build_splits(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    df = add_dedup_group(df)
    df = add_brightness_bin(df)

    eligible = df["Source group"] == "DMS"          # exclude engineered/classic rows

    df["split_random"] = "excluded"
    df.loc[eligible, "split_random"] = random_stratified_split(df[eligible], rng)

    df["split_position_holdout"] = "excluded"
    position_assign, held_out_positions = position_holdout_split(df[eligible], rng)
    df.loc[eligible, "split_position_holdout"] = position_assign

    # backbone-held-out is just leave-one-backbone-out over the eligible rows;
    # no new column needed beyond marking eligibility, since "Backbone" already
    # tells you which of the 4 folds a row belongs to
    df["split_backbone_holdout_eligible"] = eligible

    return df, held_out_positions


def summarize(df: pd.DataFrame) -> None:
    print("=== split_random ===")
    print(df["split_random"].value_counts())
    print()
    print("=== split_position_holdout ===")
    print(df["split_position_holdout"].value_counts())
    print()
    print("=== dark/bright balance across split_random ===")
    print(pd.crosstab(df["split_random"], df["brightness_bin"], normalize="index").round(3))
    print()
    print("=== dark/bright balance across split_position_holdout ===")
    print(pd.crosstab(df["split_position_holdout"], df["brightness_bin"], normalize="index").round(3))


def verify_no_leakage(df: pd.DataFrame, held_out_positions: dict) -> None:
    """Confirm no train row touches a position that was supposed to be held out."""
    mutated_positions = df["Variant name"].apply(extract_positions)
    print("\n=== leakage check: train rows touching a held-out position (should all be 0) ===")
    for backbone, held in held_out_positions.items():
        held_set = set(held["test_positions"]) | set(held["val_positions"])
        train_idx = df.index[(df["Backbone"] == backbone) & (df["split_position_holdout"] == "train")]
        n_leaked = mutated_positions.loc[train_idx].apply(lambda ps: any(p in held_set for p in ps)).sum()
        print(f"  {backbone}: {n_leaked}")

    print("\n=== leakage check: exact-duplicate sequences split across train/test/val ===")
    eligible = df[df["split_random"] != "excluded"]
    for col in ["split_random", "split_position_holdout"]:
        n_split = (eligible.groupby("dedup_group")[col].nunique() > 1).sum()
        print(f"  {col}: {n_split} dedup groups span more than one split")


def main() -> None:
    df = pd.read_pickle(IN_PATH)
    df, held_out_positions = build_splits(df)
    summarize(df)
    verify_no_leakage(df, held_out_positions)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH}, shape={df.shape}")


if __name__ == "__main__":
    main()
