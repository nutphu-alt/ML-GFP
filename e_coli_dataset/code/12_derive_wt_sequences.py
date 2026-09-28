"""
GFP variants dataset: derive each backbone's wild-type sequence.

ESM scoring needs the wild-type sequence of every backbone, but the
dataset stores only mutant sequences plus mutation lists. A variant can be
reverted to wild-type by undoing its own mutations: for each (wt, pos, mut)
the residue at `pos` is set back to `wt`.

Doing that for one variant would be enough if the data were perfect, so
instead every variant is reverted independently and the results are
compared. Agreement across tens of thousands of independent reversions is
a strong check that the mutation lists and the sequences are consistent.

Run with: python derive_wt_sequences.py
Output: output/backbone_wt.json
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"

MUT_RE = re.compile(r"([A-Z])(\d+)([A-Z])")


def revert(sequence: str, mutations: list) -> str | None:
    """Undo the listed mutations, returning the implied wild-type sequence."""
    chars = list(sequence)
    for wt, pos, mut in mutations:
        i = pos - 1
        if i < 0 or i >= len(chars) or chars[i] != mut:
            return None          # sequence disagrees with its own mutation list
        chars[i] = wt
    return "".join(chars)


def main() -> None:
    df = pd.read_pickle(OUT_DIR / "gfp_clean.pkl")
    df = df[df["Source group"] == "DMS"]

    result, report = {}, []
    for backbone, group in df.groupby("Backbone"):
        votes, bad = Counter(), 0
        for seq, name in zip(group["Protein sequence"], group["Variant name"]):
            muts = [(w, int(p), m) for w, p, m in MUT_RE.findall(str(name))]
            if not muts:
                continue
            wt = revert(seq, muts)
            if wt is None:
                bad += 1
            else:
                votes[wt] += 1

        winner, n_agree = votes.most_common(1)[0]
        total = sum(votes.values())
        result[backbone] = winner
        report.append({
            "backbone": backbone,
            "length": len(winner),
            "variants_reverted": total,
            "distinct_wt_candidates": len(votes),
            "agreement": f"{n_agree}/{total}",
            "inconsistent_rows": bad,
        })

    print(pd.DataFrame(report).to_string(index=False))
    print()
    for backbone, seq in result.items():
        print(f">{backbone}  ({len(seq)} aa)")
        print(seq[:60] + "..." if len(seq) > 60 else seq)

    (OUT_DIR / "backbone_wt.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT_DIR / 'backbone_wt.json'}")


if __name__ == "__main__":
    main()
