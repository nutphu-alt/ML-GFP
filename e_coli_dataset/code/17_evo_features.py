"""
GFP variants dataset: step 9b — cross-homolog evolutionary features.

The premise of step 9: every model plateaus near Spearman 0.44 at
never-assayed positions because nothing in the 66 dense descriptors says
anything about a specific site. BLOSUM62 knows how often Leu replaces Ile
across all proteins; it does not know that position 66 is the chromophore
tyrosine and must never change. That site-specific knowledge has to come
from outside the assay.

ESM-2 was the intended source and is unavailable offline. This is the
offline substitute: the four backbones are homologous GFPs from four
different organisms (18-83% pairwise identity), so the alignment columns
are a record of what ~hundreds of millions of years of evolution has
tolerated at each site. A column where all four agree is one evolution
has held fixed; a column where they differ is one it has varied.

Per mutation wt->mut at position p on backbone B, with column c = the
aligned column of (B, p) and H = the residues of the OTHER three
backbones at column c:

  conservation      fraction of H equal to wt        (is this site fixed?)
  mut_in_homolog    is `mut` already present in H?   (has nature made this
                                                      exact substitution?)
  evo_score         mean BLOSUM62(mut, H) - mean BLOSUM62(wt, H)
                    — the same log-odds shape as the ESM score, with a
                      3-sequence column standing in for a language model

These are then aggregated over a variant's mutations. `cons_min` (the most
conserved site the variant touches) is the feature to watch: one mutation
at an invariant site should be enough to kill the protein, which is exactly
the non-additive behaviour the additive models could never express.

NO LABEL INFORMATION IS USED. The features derive only from the four
wild-type sequences, so they are constant per (backbone, position, mutant)
and cannot leak brightness across a cross-validation fold.

Run with: python evo_features.py
Output: output/features/X_evo.npy, output/features/evo_feature_names.txt
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import importlib, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
_align_backbones = importlib.import_module("16_align_backbones")
BLOSUM62 = _align_backbones.BLOSUM62

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"

MUT_RE = re.compile(r"([A-Z])(\d+)([A-Z])")

FEATURE_NAMES = [
    "evo_score_sum", "evo_score_mean", "evo_score_min", "evo_score_max",
    "cons_mean", "cons_min", "cons_max",
    "mut_in_homolog_count", "mut_in_homolog_frac",
    "col_diversity_mean", "n_homolog_mean",
]


def blosum(a: str, b: str) -> float:
    return BLOSUM62.get((a, b), -4.0)


def build_column_index(alignment: dict):
    """column -> {backbone: residue}, skipping gaps."""
    aligned = alignment["aligned"]
    width = alignment["width"]
    index = []
    for c in range(width):
        index.append({b: seq[c] for b, seq in aligned.items() if seq[c] != "-"})
    return index


def mutation_features(backbone, wt, pos, mut, pos_to_col, col_index):
    """Return (evo_score, conservation, mut_in_homolog, diversity, n_homologs).

    Returns None when the position has no alignment column, which should not
    happen for real data and is counted as a skip if it does.
    """
    col = pos_to_col[backbone].get(str(pos))
    if col is None:
        return None
    residues = col_index[col]
    homologs = [r for b, r in residues.items() if b != backbone]
    if not homologs:
        return 0.0, 0.0, 0.0, 1.0, 0.0

    wt_fit = sum(blosum(wt, h) for h in homologs) / len(homologs)
    mut_fit = sum(blosum(mut, h) for h in homologs) / len(homologs)
    conservation = sum(h == wt for h in homologs) / len(homologs)
    in_homolog = float(mut in homologs)
    diversity = float(len(set(residues.values())))
    return mut_fit - wt_fit, conservation, in_homolog, diversity, float(len(homologs))


def build_features(meta, alignment, wt_sequences):
    pos_to_col = alignment["pos_to_col"]
    col_index = build_column_index(alignment)

    X = np.zeros((len(meta), len(FEATURE_NAMES)), dtype=np.float32)
    skipped_rows, residue_mismatch, no_mutations = 0, 0, 0

    for row, (backbone, name) in enumerate(
            zip(meta["Backbone"], meta["Variant name"])):
        scores, cons, seen, div, nhom = [], [], [], [], []
        for wt, pos_s, mut in MUT_RE.findall(str(name)):
            pos = int(pos_s)
            seq = wt_sequences.get(backbone, "")
            # same residue-consistency check esm_scores.py applies: the
            # stated wild-type residue must match the derived WT sequence
            if not (1 <= pos <= len(seq)) or seq[pos - 1] != wt:
                residue_mismatch += 1
                continue
            out = mutation_features(backbone, wt, pos, mut, pos_to_col, col_index)
            if out is None:
                skipped_rows += 1
                continue
            s, c, m, d, n = out
            scores.append(s); cons.append(c); seen.append(m)
            div.append(d); nhom.append(n)

        if not scores:
            no_mutations += 1
            continue
        s = np.array(scores, dtype=np.float32)
        c = np.array(cons, dtype=np.float32)
        m = np.array(seen, dtype=np.float32)
        X[row] = [
            s.sum(), s.mean(), s.min(), s.max(),
            c.mean(), c.min(), c.max(),
            m.sum(), m.mean(),
            float(np.mean(div)), float(np.mean(nhom)),
        ]

    print(f"rows with no scorable mutation: {no_mutations}")
    print(f"mutations skipped, residue mismatch: {residue_mismatch}")
    print(f"mutations skipped, no alignment column: {skipped_rows}")
    return X


def main() -> None:
    alignment = json.loads((OUT_DIR / "backbone_alignment.json").read_text())
    wt_sequences = json.loads((OUT_DIR / "backbone_wt.json").read_text())
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    print(f"variants: {len(meta)}, alignment width: {alignment['width']}")

    X = build_features(meta, alignment, wt_sequences)
    scored = int((X != 0).any(axis=1).sum())
    print(f"\nX_evo: {X.shape}, {scored} rows scored ({scored / len(meta):.1%})")
    print(pd.DataFrame(X, columns=FEATURE_NAMES).describe().round(3).to_string())

    # a quick sanity read: variants touching a fully conserved site should be
    # dimmer on average than those touching a fully variable one
    y = np.load(FEAT_DIR / "y.npy")
    cons_min = X[:, FEATURE_NAMES.index("cons_min")]
    for lo, hi, label in [(0.99, 1.01, "all 3 homologs share the WT residue"),
                          (0.65, 0.68, "2 of 3 share it"),
                          (0.32, 0.35, "1 of 3 shares it"),
                          (-0.01, 0.01, "none share it")]:
        mask = (cons_min >= lo) & (cons_min <= hi)
        if mask.sum():
            print(f"  cons_min ~ {label:38s} n={mask.sum():6d} "
                  f"mean brightness={y[mask].mean():.3f}")

    np.save(FEAT_DIR / "X_evo.npy", X)
    (FEAT_DIR / "evo_feature_names.txt").write_text("\n".join(FEATURE_NAMES))
    print(f"\nwrote {FEAT_DIR / 'X_evo.npy'}")


if __name__ == "__main__":
    main()
