"""
GFP variants dataset: merge, clean, tag, and EDA.

Reproduces step 1 of the GFP modeling plan:
  - merge the 15 source .xlsx files
  - drop unusable rows
  - tag each row with its backbone and source group
  - save a modeling-ready CSV
  - produce two EDA plots

Run with: python clean_gfp_data.py   (from anywhere -- paths are resolved
relative to this script's location, not the current working directory)
"""

import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # no display in this environment; write plots straight to file
import matplotlib.pyplot as plt
import pandas as pd

# this file lives in e_coli_dataset/code/, so its parent's parent is the e_coli_dataset/ folder
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "data"      # the 15 source .xlsx files
OUT_DIR = BASE_DIR / "output"    # where cleaned outputs get written

KNOWN_BACKBONES = {"avGFP", "amacGFP", "cgreGFP", "ppluGFP2"}
MUTATION_PATTERN = re.compile(r"[A-Z]\d+[A-Z]")             # e.g. matches "I32V" inside a variant name


def load_and_merge(src_dir: str) -> pd.DataFrame:
    """Read all 15 GFP_variants_part*.xlsx files and stack them into one dataframe."""
    files = sorted(glob.glob(f"{src_dir}/GFP_variants_part*.xlsx"))
    frames = [pd.read_excel(f, sheet_name="GFP variants") for f in files]
    return pd.concat(frames, ignore_index=True)


def tag_backbone(row: pd.Series) -> str:
    """Work out which GFP backbone a row belongs to, from its variant name / classification."""
    name = row["Variant name"]
    first_token = name.split()[0] if isinstance(name, str) and name.split() else name
    if first_token in KNOWN_BACKBONES:
        return first_token
    cls = row["Variant classification"]
    if isinstance(cls, str) and "natural" in cls.lower():
        return "natural_other"    # one-off natural FP, not part of a DMS backbone family
    return "avGFP"                # the ~12 classic engineered variants are all avGFP-lineage


def tag_source_group(analyzing_method: str) -> str:
    """Bucket each row by assay/scale, not by the classification label.

    This matters because brightness values are only comparable within the
    same assay. Several rows carry a "Wild-type" or "Wild-type (library
    parent)" classification but were actually measured by the same
    FACS-seq assay as the DMS libraries (same MFI/brightness scale) -- so
    grouping by classification text alone mis-files them. Grouping by the
    analyzing method instead correctly keeps them with the DMS-scale data.
    """
    if not isinstance(analyzing_method, str):
        return "other"
    m = analyzing_method.lower()
    if "facs-seq" in m:
        return "DMS"
    if "colony image" in m:
        return "ML-designed multi-mutant"
    return "engineered/classic"   # spectrofluorometer / EC×QY scale -- not comparable to DMS rows


def count_mutations(name: str) -> int:
    """Count point mutations (e.g. I32V) mentioned in a variant name."""
    if not isinstance(name, str):
        return 0
    return len(MUTATION_PATTERN.findall(name))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unusable rows and add derived columns."""
    df = df[~df["Protein sequence"].str.contains("X", na=False)].copy()   # unresolved residues
    df = df[df["Brightness value"].notna()].copy()                       # rows with no usable label

    df["Backbone"] = df.apply(tag_backbone, axis=1)
    df["Source group"] = df["Analyzing method"].apply(tag_source_group)
    df["n_mutations"] = df["Variant name"].apply(count_mutations)
    df["seq_len"] = df["Protein sequence"].str.len()
    return df


def save_clean_csv(df: pd.DataFrame, out_path: str) -> None:
    cols = [
        "Protein sequence", "Variant name", "Backbone", "Source group", "n_mutations",
        "Brightness value", "Mean fluorescence intensity (MFI)", "Variant classification",
        "Cell type", "Expression system", "Analyzing method", "Reference",
    ]
    df[cols].reset_index(drop=True).to_csv(out_path, index=False)


def plot_eda(df: pd.DataFrame, out_path: str) -> None:
    """Two side-by-side histograms: brightness and mutation-count, split by backbone."""
    dms = df[df["Source group"] == "DMS"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for backbone, group in dms.groupby("Backbone"):
        axes[0].hist(group["Brightness value"], bins=80, alpha=0.5, label=backbone, density=True)
    axes[0].set_xlabel("Brightness value (fold-WT)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Brightness distribution by backbone (DMS rows)")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, 2.5)

    for backbone, group in dms.groupby("Backbone"):
        axes[1].hist(group["n_mutations"], bins=range(1, 12), alpha=0.5, label=backbone, density=True)
    axes[1].set_xlabel("# mutations from WT")
    axes[1].set_title("Mutation-count distribution by backbone")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_and_merge(SRC_DIR)
    print("Merged shape:", raw.shape)

    cleaned = clean(raw)
    print("Cleaned shape:", cleaned.shape)
    print(cleaned["Backbone"].value_counts())
    print(cleaned["Source group"].value_counts())

    # the .pkl keeps dtypes intact and is what make_splits.py (step 2) reads;
    # the .csv is the human-readable / portable copy of the same table
    cleaned.to_pickle(OUT_DIR / "gfp_clean.pkl")
    save_clean_csv(cleaned, OUT_DIR / "gfp_variants_clean.csv")
    plot_eda(cleaned, OUT_DIR / "eda_distributions.png")
    print("Done: gfp_clean.pkl, gfp_variants_clean.csv and eda_distributions.png written.")


if __name__ == "__main__":
    main()
