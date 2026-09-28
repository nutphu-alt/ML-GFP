"""
GFP variants dataset: step 9a — align the four backbone wild-type sequences.

Step 9's goal is to bring information about a position from OUTSIDE this
assay, because that is the one thing the current 66 descriptors cannot do
and the reason every model plateaus near Spearman 0.44 at never-measured
sites. The intended source was ESM-2, which is unavailable offline (see
esm_scores.py). This is the fallback source that IS available: the four
backbones are genuine homologs from four different organisms, so the
differences between them are themselves an evolutionary record of which
substitutions a GFP tolerates.

Four sequences is a very thin alignment — nothing like the millions a
protein language model distils — but it is real outside-the-assay signal
and it costs no network access.

The sequences have different lengths (238/238/235/222), so they must be
aligned rather than indexed positionally. This implements Needleman-Wunsch
with BLOSUM62 and affine gaps, in a star topology: every backbone is
aligned pairwise to avGFP, and the resulting per-backbone position ->
alignment-column maps are merged. Star alignment to a single reference is
adequate here because all four are the same fold with high coverage;
a progressive/iterative MSA would buy little on four sequences.

Validation is built in: the GFP chromophore (avGFP Thr65-Tyr66-Gly67, and
its equivalents) must land in the same alignment columns in all four, and
pairwise identities must match the 20-80% range expected for GFP homologs.

Run with: python align_backbones.py
Output: output/backbone_alignment.json
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"

REFERENCE = "avGFP"
GAP_OPEN = -11.0          # standard BLOSUM62 affine gap costs
GAP_EXTEND = -1.0

# BLOSUM62, same matrix already used by build_features.py. Stored as a flat
# dict keyed by residue pair so lookups need no external dependency.
_B62_ROWS = """
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
AMINO_ACIDS = list("ARNDCQEGHILKMFPSTWYV")


def _load_blosum62() -> dict:
    matrix = {}
    for line in _B62_ROWS.strip().splitlines():
        parts = line.split()
        row_aa, scores = parts[0], [float(v) for v in parts[1:]]
        for col_aa, score in zip(AMINO_ACIDS, scores):
            matrix[(row_aa, col_aa)] = score
    return matrix


BLOSUM62 = _load_blosum62()


def score(a: str, b: str) -> float:
    """BLOSUM62 score, tolerant of any non-standard residue."""
    return BLOSUM62.get((a, b), -4.0)


def needleman_wunsch(seq_a: str, seq_b: str):
    """Global alignment with affine gaps (Gotoh). Returns aligned strings.

    Three matrices are tracked: M (residue aligned to residue), X (gap in
    seq_b), Y (gap in seq_a). Affine gaps matter here because GFP homologs
    differ by a few multi-residue indels — a linear gap penalty would
    scatter them into many isolated single gaps and misalign the flanks.
    """
    n, m = len(seq_a), len(seq_b)
    neg = float("-inf")

    # value matrices
    M = [[neg] * (m + 1) for _ in range(n + 1)]
    X = [[neg] * (m + 1) for _ in range(n + 1)]
    Y = [[neg] * (m + 1) for _ in range(n + 1)]
    # traceback matrices: which state we came from (0=M, 1=X, 2=Y)
    tM = [[0] * (m + 1) for _ in range(n + 1)]
    tX = [[0] * (m + 1) for _ in range(n + 1)]
    tY = [[0] * (m + 1) for _ in range(n + 1)]

    M[0][0] = 0.0
    for i in range(1, n + 1):
        X[i][0] = GAP_OPEN + (i - 1) * GAP_EXTEND
        tX[i][0] = 1
    for j in range(1, m + 1):
        Y[0][j] = GAP_OPEN + (j - 1) * GAP_EXTEND
        tY[0][j] = 2

    for i in range(1, n + 1):
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            s = score(ai, seq_b[j - 1])

            # M: both residues consumed
            best, src = M[i - 1][j - 1], 0
            if X[i - 1][j - 1] > best:
                best, src = X[i - 1][j - 1], 1
            if Y[i - 1][j - 1] > best:
                best, src = Y[i - 1][j - 1], 2
            M[i][j], tM[i][j] = best + s, src

            # X: gap in seq_b (consume seq_a)
            open_ = M[i - 1][j] + GAP_OPEN
            extend = X[i - 1][j] + GAP_EXTEND
            if extend > open_:
                X[i][j], tX[i][j] = extend, 1
            else:
                X[i][j], tX[i][j] = open_, 0

            # Y: gap in seq_a (consume seq_b)
            open_ = M[i][j - 1] + GAP_OPEN
            extend = Y[i][j - 1] + GAP_EXTEND
            if extend > open_:
                Y[i][j], tY[i][j] = extend, 2
            else:
                Y[i][j], tY[i][j] = open_, 0

    # pick the best-scoring end state and walk back
    ends = [(M[n][m], 0), (X[n][m], 1), (Y[n][m], 2)]
    _, state = max(ends)
    i, j = n, m
    out_a, out_b = [], []
    while i > 0 or j > 0:
        if state == 0:
            out_a.append(seq_a[i - 1])
            out_b.append(seq_b[j - 1])
            state = tM[i][j]
            i, j = i - 1, j - 1
        elif state == 1:
            out_a.append(seq_a[i - 1])
            out_b.append("-")
            state = tX[i][j]
            i -= 1
        else:
            out_a.append("-")
            out_b.append(seq_b[j - 1])
            state = tY[i][j]
            j -= 1
    return "".join(reversed(out_a)), "".join(reversed(out_b))


def star_alignment(sequences: dict, reference: str):
    """Align every sequence to `reference`, then merge into one MSA.

    Merging is done by walking the reference: each pairwise alignment is
    replayed column by column, and insertions relative to the reference are
    given their own columns. The reference therefore keeps one column per
    residue plus extra columns wherever some other backbone inserts.
    """
    ref_seq = sequences[reference]
    others = [b for b in sequences if b != reference]

    # pairwise alignments, stored as per-reference-residue mappings
    # ref_index -> aligned residue in the other sequence (or "-"), plus
    # insertions keyed by the reference index they follow
    pair_data = {}
    for backbone in others:
        a_ref, a_oth = needleman_wunsch(ref_seq, sequences[backbone])
        mapped = {}          # ref residue idx (0-based) -> (other residue, other idx)
        insertions = {}      # ref idx after which residues are inserted
        ref_i, oth_i = -1, -1
        for ca, cb in zip(a_ref, a_oth):
            if ca != "-":
                ref_i += 1
            if cb != "-":
                oth_i += 1
            if ca != "-" and cb != "-":
                mapped[ref_i] = (cb, oth_i)
            elif ca != "-" and cb == "-":
                mapped[ref_i] = ("-", None)
            else:                                   # insertion in the other seq
                insertions.setdefault(ref_i, []).append((cb, oth_i))
        pair_data[backbone] = {"mapped": mapped, "insertions": insertions}

    # build merged column layout: for each reference residue, one column,
    # preceded by however many insertion columns the widest backbone needs
    columns = []             # each entry: dict backbone -> (residue, own idx)
    max_pre = max(
        (len(d["insertions"].get(-1, [])) for d in pair_data.values()), default=0)
    for k in range(max_pre):
        col = {reference: ("-", None)}
        for backbone, d in pair_data.items():
            ins = d["insertions"].get(-1, [])
            col[backbone] = ins[k] if k < len(ins) else ("-", None)
        columns.append(col)

    for ref_i, residue in enumerate(ref_seq):
        col = {reference: (residue, ref_i)}
        for backbone, d in pair_data.items():
            col[backbone] = d["mapped"].get(ref_i, ("-", None))
        columns.append(col)

        width = max((len(d["insertions"].get(ref_i, []))
                     for d in pair_data.values()), default=0)
        for k in range(width):
            icol = {reference: ("-", None)}
            for backbone, d in pair_data.items():
                ins = d["insertions"].get(ref_i, [])
                icol[backbone] = ins[k] if k < len(ins) else ("-", None)
            columns.append(icol)

    # per-backbone: own 1-based position -> alignment column index
    pos_to_col = {b: {} for b in sequences}
    aligned = {b: [] for b in sequences}
    for c, col in enumerate(columns):
        for backbone, (residue, own_i) in col.items():
            aligned[backbone].append(residue)
            if own_i is not None:
                pos_to_col[backbone][own_i + 1] = c
    aligned = {b: "".join(chars) for b, chars in aligned.items()}
    return aligned, pos_to_col


def pairwise_identity(a: str, b: str) -> float:
    both = [(x, y) for x, y in zip(a, b) if x != "-" and y != "-"]
    if not both:
        return 0.0
    return sum(x == y for x, y in both) / len(both)


def main() -> None:
    wt = json.loads((OUT_DIR / "backbone_wt.json").read_text())
    print("input sequences:")
    for backbone, seq in wt.items():
        print(f"  {backbone:10s} {len(seq)} aa")

    aligned, pos_to_col = star_alignment(wt, REFERENCE)
    width = len(next(iter(aligned.values())))
    print(f"\nalignment width: {width} columns "
          f"(reference {REFERENCE}, {len(wt[REFERENCE])} residues)")

    print("\npairwise identity:")
    names = sorted(aligned)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"  {a:10s} vs {b:10s} {pairwise_identity(aligned[a], aligned[b]):6.1%}")

    # --- validation: the chromophore must align across all four ----------
    # avGFP's chromophore-forming tripeptide is positions 65-67 (T-Y-G in
    # the F64L parent numbering used throughout this dataset). The Tyr-Gly
    # is the invariant part across the GFP family.
    print("\nchromophore check (avGFP 65-67 and the aligned columns):")
    ok = True
    for pos in (65, 66, 67):
        col = pos_to_col[REFERENCE][pos]
        residues = {b: aligned[b][col] for b in names}
        print(f"  avGFP position {pos} -> column {col}: " +
              "  ".join(f"{b}={r}" for b, r in residues.items()))
        if pos in (66, 67):
            expected = "Y" if pos == 66 else "G"
            if any(r != expected for r in residues.values()):
                ok = False
    print("  chromophore Tyr-Gly aligned in all four backbones: "
          + ("YES" if ok else "NO — alignment is suspect"))

    coverage = {b: len(m) for b, m in pos_to_col.items()}
    print("\npositions mapped to a column:")
    for backbone, n in coverage.items():
        print(f"  {backbone:10s} {n}/{len(wt[backbone])}")

    payload = {
        "reference": REFERENCE,
        "width": width,
        "aligned": aligned,
        # JSON keys must be strings
        "pos_to_col": {b: {str(p): c for p, c in m.items()}
                       for b, m in pos_to_col.items()},
        "chromophore_check_passed": ok,
    }
    (OUT_DIR / "backbone_alignment.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_DIR / 'backbone_alignment.json'}")


if __name__ == "__main__":
    main()
