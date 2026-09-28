"""
GFP variants dataset: step 11b — ESM-2 *embeddings*, not just zero-shot scores.

Step 11 tested the zero-shot log-odds formulation and got a clean negative:
ESM-2 650M does not beat the cross-homolog evo features, and adds nothing
significant on top of them (+0.0085, CI [-0.0028, +0.0199]).

But the project's open item asked for ESM-2 *embeddings*, and that is a
different signal. A log-odds score says "how likely is this substitution".
An embedding says "what kind of site is this" — buried or exposed, helix or
strand, near the chromophore or not — which is site-specific structural
context the log-odds scalar collapses away.

The obvious formulation (embed every variant sequence) needs 141,144 forward
passes and is not affordable. This uses the same trick as the rest of the
project: embed only the four WILD-TYPE sequences, one forward pass each, take
the per-residue hidden states, and let a variant inherit the embeddings of the
positions it mutates. Cost is 4 forward passes, and the features are constant
per (backbone, position), so — like the evo features — they carry NO label
information and cannot leak across a fold.

1280 dims per position is far too wide for 141k rows next to 66 dense
features, so the per-position matrix is PCA-reduced. The PCA is fitted on the
wild-type residue embeddings only, which involves no labels and no train/test
split, so it is fold-safe.

Per variant: the mean, and the element-wise min/max spread, of its mutated
positions' reduced embeddings.

Run with: python esm_embed.py [--hf-model ...] [--dims 16]
Output: output/features/X_esmemb{suffix}.npy, esmemb_feature_names.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"

import re
MUT_RE = re.compile(r"([A-Z])(\d+)([A-Z])")


def wt_embeddings(wt_sequences, hf_model, cache: Path):
    """{backbone: (L, H)} per-residue hidden states of the wild-type."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    cache.mkdir(parents=True, exist_ok=True)
    tag = hf_model.split("/")[-1]
    need = [b for b in wt_sequences if not (cache / f"emb_{tag}_{b}.npy").exists()]
    if need:
        print(f"loading {hf_model} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(hf_model)
        model = AutoModel.from_pretrained(hf_model)
        model.eval()
        torch.set_grad_enabled(False)
        for b in need:
            seq = wt_sequences[b]
            enc = tok(seq, return_tensors="pt")
            assert enc["input_ids"].shape[1] == len(seq) + 2
            h = model(**enc).last_hidden_state[0, 1:len(seq) + 1].numpy()
            np.save(cache / f"emb_{tag}_{b}.npy", h.astype(np.float32))
            print(f"  {b}: {h.shape}", flush=True)
    return {b: np.load(cache / f"emb_{tag}_{b}.npy") for b in wt_sequences}


def reduce_embeddings(emb: dict, dims: int):
    """PCA on the pooled wild-type residue embeddings. No labels involved."""
    stacked = np.vstack([emb[b] for b in sorted(emb)])
    pca = PCA(n_components=dims, random_state=0).fit(stacked)
    print(f"PCA {stacked.shape[1]} -> {dims} dims, "
          f"explained variance {pca.explained_variance_ratio_.sum():.3f}")
    return {b: pca.transform(v).astype(np.float32) for b, v in emb.items()}


def build_features(meta, reduced, wt_sequences, dims):
    names = ([f"esmemb_mean_{i}" for i in range(dims)]
             + [f"esmemb_spread_{i}" for i in range(dims)])
    X = np.zeros((len(meta), 2 * dims), dtype=np.float32)
    skipped = 0
    for row, (backbone, name) in enumerate(
            zip(meta["Backbone"], meta["Variant name"])):
        R = reduced.get(backbone)
        seq = wt_sequences.get(backbone, "")
        if R is None:
            continue
        vecs = []
        for wt, pos_s, _mut in MUT_RE.findall(str(name)):
            p = int(pos_s) - 1
            # same residue-consistency check the rest of the project applies
            if not (0 <= p < len(R)) or seq[p] != wt:
                skipped += 1
                continue
            vecs.append(R[p])
        if vecs:
            V = np.array(vecs, dtype=np.float32)
            X[row, :dims] = V.mean(axis=0)
            X[row, dims:] = V.max(axis=0) - V.min(axis=0)
    if skipped:
        print(f"note: {skipped} mutations skipped (position or residue mismatch)")
    return X, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--dims", type=int, default=16)
    ap.add_argument("--suffix", default="_650M")
    args = ap.parse_args()

    wt_sequences = json.loads((OUT_DIR / "backbone_wt.json").read_text())
    meta = pd.read_csv(FEAT_DIR / "meta.csv")

    emb = wt_embeddings(wt_sequences, args.hf_model, OUT_DIR / "esm_cache")
    reduced = reduce_embeddings(emb, args.dims)
    X, names = build_features(meta, reduced, wt_sequences, args.dims)

    scored = int((X != 0).any(axis=1).sum())
    print(f"X_esmemb: {X.shape}, {scored} rows scored ({scored / len(meta):.1%})")
    np.save(FEAT_DIR / f"X_esmemb{args.suffix}.npy", X)
    (FEAT_DIR / "esmemb_feature_names.txt").write_text("\n".join(names))
    print(f"wrote {FEAT_DIR / f'X_esmemb{args.suffix}.npy'}")


if __name__ == "__main__":
    main()
