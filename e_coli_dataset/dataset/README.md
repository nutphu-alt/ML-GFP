# GFP variant dataset for sequence → fluorescence machine learning

15 Excel files, 141,469 rows total, ≤10,000 rows each, sorted by MFI (descending; blank-MFI rows last).
Every file also carries a **Notes & sources** sheet with the same information below.

## Columns

| # | Column | Notes |
|---|---|---|
| 1 | Protein sequence | Full-length amino-acid sequence |
| 2 | Variant name | Backbone + mutation list, standard numbering (Met1 counted) |
| 3 | Mean fluorescence intensity (MFI) | Source's own arbitrary units — **not comparable across backbones** |
| 4 | % positive cells | Almost always blank (see below) |
| 5 | Brightness value | **Use this as the ML label** — MFI ÷ wild-type MFI of the same backbone (fold-WT) |
| 6 | Variant classification | Wild-type / directed evolution / rational-ML design / monomeric |
| 7 | Cell type | |
| 8 | Expression system | |
| 9 | Analyzing method | Includes the units/scale for that row |
| 10 | Reference | Full citation + DOI |

## Contents

| Source | Rows | Assay |
|---|---|---|
| avGFP deep mutational scan — Sarkisyan et al. 2016 (re-filtered release from Gonzalez Somermeyer et al. 2022) | 51,715 | E. coli, FACS-seq |
| amacGFP / ppluGFP2 / cgreGFP deep mutational scans — Gonzalez Somermeyer et al. 2022 | 89,429 | E. coli, FACS-seq |
| ML-designed multi-mutants, experimentally validated — same 2022 study | 264 | E. coli, colony fluorescence microscopy |
| Natural (wild-type) green FPs, full sequences + primary references | 49 | Spectrofluorometry |
| Classic engineered avGFP-lineage variants (EGFP, sfGFP, Emerald, GFPuv, mEGFP, msfGFP, GFPmut2/3, Sapphire, …) | 12 | Molecular brightness (EC × QY) |

## Wild-type reference points (for normalising column 5)

| Backbone | Wild-type MFI |
|---|---|
| avGFP (F64L parent) | 5,238.6 |
| amacGFP | 9,348.3 |
| cgreGFP | 31,398.9 |
| ppluGFP2 | 16,819.9 |

## Data-quality decisions

- **Sequences were reconstructed, not copied.** Each source ships mutation lists, not sequences. Every mutation was validated against the parent residue at that position before being applied; any genotype that failed was dropped.
- **6,806 of 147,950 source genotypes dropped (4.6 %)**: 6,231 encode indels or premature stops (source notation `.` and `*`) and so have no well-defined full-length protein; 575 had a stated wild-type residue that did not match the parent.
- **Numbering was re-indexed.** The source files use a zero-shifted convention; column 2 uses standard mature-protein numbering (avGFP S65T, F64L, etc.).
- Three natural FPs (cgigGFP, scubGFP1, scubGFP2) contain an `X` at an unresolved position in the published sequence — filter these out before training.
- The 12 engineered-variant sequences were built from wild-type avGFP using each paper's published mutation set; all mutation lists validated cleanly against avGFP.

## Two known gaps

1. **% positive cells is essentially empty.** FACS-seq and colony-imaging assays don't report a percent-positive gate, and mammalian-cell GFP papers rarely publish it in machine-readable form. Filling this column meaningfully needs manual extraction from individual figures.
2. **FPbase was not harvested.** Its API was unreachable from the sandbox and timed out repeatedly through the web fetcher. Its ~250 curated green FP entries would substantially expand the named-engineered-variant portion.

## Suggested modelling note

Train on column 5 (fold-WT brightness), not column 3. Consider a per-backbone fixed effect or separate models — the four DMS libraries were measured on different instruments and their raw MFI scales are unrelated.

## References

- Sarkisyan KS et al. *Local fitness landscape of the green fluorescent protein.* Nature 533:397–401 (2016). doi:10.1038/nature17995
- Gonzalez Somermeyer L et al. *Heterogeneity of the GFP fitness landscape and data-driven protein design.* eLife 11:e75842 (2022). doi:10.7554/eLife.75842
- Prasher DC et al. Gene 111:229–233 (1992) · Heim R, Cubitt AB, Tsien RY. Nature 373:663–664 (1995) · Cormack BP et al. Gene 173:33–38 (1996) · Crameri A et al. Nat Biotechnol 14:315–319 (1996) · Waldo GS et al. Nat Biotechnol 17:691–695 (1999) · Pédelacq JD et al. Nat Biotechnol 24:79–88 (2006) · Zacharias DA et al. Science 296:913–916 (2002) · Zapata-Hommer O, Griesbeck O. BMC Biotechnol 3:5 (2003)
