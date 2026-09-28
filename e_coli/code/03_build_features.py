"""
GFP variants dataset: step 3a — build feature matrices.

Two complementary feature blocks are produced, because they answer
different questions:

  SPARSE block ("memorisation" features)
      One binary column per (backbone, position, mutant amino acid).
      Keeping the backbone in the key means a linear model fitted on
      these is effectively four independent per-backbone models sharing
      one regulariser -- which is what we want, since the duplicate-
      sequence analysis showed brightness is NOT comparable across
      libraries (same protein read 1.000 fold-WT in the avGFP library
      and 0.245 in the amacGFP library).
      These columns cannot say anything about a position never seen in
      training, so on the position-holdout split they are expected to
      collapse. That is the point of having them.

  DENSE block ("transferable" features)
      Per-row aggregates that carry meaning at a position the model has
      never seen: BLOSUM62 substitution scores, hydropathy/volume/charge
      deltas, which amino acids were mutated from and to, mutation count,
      and backbone one-hot. This is the block that has any chance of
      extrapolating to held-out positions.

Run with: python build_features.py [--limit N]
  --limit N   only use the first N eligible rows (smoke test)

Input : output/gfp_variants_split.csv  (or output/gfp_clean.pkl + splits)
Output: output/features/{X_sparse.npz, X_dense.npy, y.npy, meta.csv,
                        sparse_feature_names.txt, dense_feature_names.txt}
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"

SPLIT_CSV = OUT_DIR / "gfp_variants_split.csv"

# <wt_aa><position><mutant_aa>, e.g. "I32V"
MUTATION_RE = re.compile(r"([A-Z])(\d+)([A-Z])")

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydropathy
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# side-chain volume, cubic angstroms (Zamyatnin)
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}

# formal charge at physiological pH
CHARGE = {aa: 0.0 for aa in AMINO_ACIDS}
CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5})

POLAR = set("STNQYCHKRDEW")

# BLOSUM62, upper-triangular by row; symmetric lookup built below.
_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_ROWS = [
    " 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0",
    "-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3",
    "-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3",
    "-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3",
    " 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1",
    "-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2",
    "-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2",
    " 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3",
    "-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3",
    "-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3",
    "-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1",
    "-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2",
    "-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1",
    "-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1",
    "-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2",
    " 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2",
    " 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0",
    "-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3",
    "-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1",
    " 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4",
]
BLOSUM62 = {}
for _i, _row in enumerate(_BLOSUM62_ROWS):
    for _j, _val in enumerate(_row.split()):
        BLOSUM62[(_BLOSUM62_ORDER[_i], _BLOSUM62_ORDER[_j])] = float(_val)


def parse_mutations(variant_name: str) -> list[tuple[str, int, str]]:
    """Return [(wt_aa, position, mutant_aa), ...] parsed from a variant name."""
    if not isinstance(variant_name, str):
        return []
    return [(w, int(p), m) for w, p, m in MUTATION_RE.findall(variant_name)]


def load_rows(limit: int | None) -> pd.DataFrame:
    """Load the split dataset, keeping only rows eligible for modeling."""
    df = pd.read_csv(SPLIT_CSV)
    df = df[df["Source group"] == "DMS"].reset_index(drop=True)
    if limit:
        df = df.head(limit).copy()
    return df


def build_sparse_block(df: pd.DataFrame, mutations: pd.Series):
    """Binary indicator per (backbone, position, mutant_aa).

    Built with explicit row/col/value triplets in COO form, which is the
    cheapest way to assemble a matrix this sparse (~3 nonzeros per row).
    """
    feature_index: dict[tuple[str, int, str], int] = {}
    rows, cols = [], []

    for row_i, (backbone, muts) in enumerate(zip(df["Backbone"], mutations)):
        for _wt, pos, mut in muts:
            key = (backbone, pos, mut)
            col = feature_index.get(key)
            if col is None:
                col = len(feature_index)
                feature_index[key] = col
            rows.append(row_i)
            cols.append(col)

    data = np.ones(len(rows), dtype=np.float32)
    X = sparse.coo_matrix(
        (data, (rows, cols)), shape=(len(df), len(feature_index)), dtype=np.float32
    ).tocsr()

    names = [None] * len(feature_index)
    for (backbone, pos, mut), col in feature_index.items():
        names[col] = f"{backbone}:{pos}{mut}"
    return X, names


def build_dense_block(df: pd.DataFrame, mutations: pd.Series):
    """Compact features that still mean something at an unseen position."""
    backbones = sorted(df["Backbone"].unique())
    aa_to_i = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    names = (
        ["n_mutations"]
        + [f"backbone={b}" for b in backbones]
        + ["blosum_sum", "blosum_mean", "blosum_min", "blosum_max"]
        + ["dhydro_sum", "dhydro_absmean", "dhydro_min", "dhydro_max"]
        + ["dvol_sum", "dvol_absmean", "dvol_min", "dvol_max"]
        + ["dcharge_sum", "dcharge_absmean"]
        + ["n_to_polar", "n_from_polar", "n_to_proline", "n_from_glycine"]
        + ["pos_mean_frac", "pos_min_frac", "pos_max_frac"]
        + [f"to_{aa}" for aa in AMINO_ACIDS]
        + [f"from_{aa}" for aa in AMINO_ACIDS]
    )

    X = np.zeros((len(df), len(names)), dtype=np.float32)
    backbone_col = {b: 1 + i for i, b in enumerate(backbones)}
    base = 1 + len(backbones)
    to_base = len(names) - 40
    from_base = len(names) - 20

    seq_lens = df["Protein sequence"].str.len().to_numpy()

    for i, (backbone, muts) in enumerate(zip(df["Backbone"], mutations)):
        X[i, 0] = len(muts)
        X[i, backbone_col[backbone]] = 1.0
        if not muts:
            continue

        seq_len = max(int(seq_lens[i]), 1)
        blosum, dhydro, dvol, dcharge, positions = [], [], [], [], []

        for wt, pos, mut in muts:
            blosum.append(BLOSUM62.get((wt, mut), 0.0))
            dhydro.append(HYDROPATHY.get(mut, 0.0) - HYDROPATHY.get(wt, 0.0))
            dvol.append(VOLUME.get(mut, 0.0) - VOLUME.get(wt, 0.0))
            dcharge.append(CHARGE.get(mut, 0.0) - CHARGE.get(wt, 0.0))
            positions.append(pos / seq_len)

            if mut in POLAR:
                X[i, base + 14] += 1
            if wt in POLAR:
                X[i, base + 15] += 1
            if mut == "P":
                X[i, base + 16] += 1
            if wt == "G":
                X[i, base + 17] += 1
            if mut in aa_to_i:
                X[i, to_base + aa_to_i[mut]] += 1
            if wt in aa_to_i:
                X[i, from_base + aa_to_i[wt]] += 1

        blosum = np.array(blosum); dhydro = np.array(dhydro)
        dvol = np.array(dvol); dcharge = np.array(dcharge)
        positions = np.array(positions)

        X[i, base + 0:base + 4] = [blosum.sum(), blosum.mean(), blosum.min(), blosum.max()]
        X[i, base + 4:base + 8] = [dhydro.sum(), np.abs(dhydro).mean(), dhydro.min(), dhydro.max()]
        X[i, base + 8:base + 12] = [dvol.sum(), np.abs(dvol).mean(), dvol.min(), dvol.max()]
        X[i, base + 12:base + 14] = [dcharge.sum(), np.abs(dcharge).mean()]
        X[i, base + 18:base + 21] = [positions.mean(), positions.min(), positions.max()]

    return X, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N rows (smoke test)")
    args = parser.parse_args()

    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_rows(args.limit)
    print(f"eligible rows: {len(df)}")

    mutations = df["Variant name"].apply(parse_mutations)
    print(f"mutations parsed, mean per row: {mutations.apply(len).mean():.2f}")

    X_sparse, sparse_names = build_sparse_block(df, mutations)
    print(f"sparse block: {X_sparse.shape}, nnz={X_sparse.nnz}, "
          f"density={X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]):.2e}")

    X_dense, dense_names = build_dense_block(df, mutations)
    print(f"dense block : {X_dense.shape}")

    y = df["Brightness value"].to_numpy(dtype=np.float32)
    meta = df[["Backbone", "split_random", "split_position_holdout", "Variant name"]].copy()

    sparse.save_npz(FEAT_DIR / "X_sparse.npz", X_sparse)
    np.save(FEAT_DIR / "X_dense.npy", X_dense)
    np.save(FEAT_DIR / "y.npy", y)
    meta.to_csv(FEAT_DIR / "meta.csv", index=False)
    (FEAT_DIR / "sparse_feature_names.txt").write_text("\n".join(sparse_names))
    (FEAT_DIR / "dense_feature_names.txt").write_text("\n".join(dense_names))

    print(f"\nwrote features to {FEAT_DIR}")


if __name__ == "__main__":
    main()
