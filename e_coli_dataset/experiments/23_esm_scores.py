"""
GFP variants dataset: step 6 — ESM-2 zero-shot variant scores.

Every model so far plateaus near Spearman 0.44 at never-assayed positions,
because nothing in the feature set describes a site the training data has
not seen. A protein language model does: its log-probabilities come from
evolutionary data rather than this assay, so it has an opinion about
positions that were never measured.

This uses the standard zero-shot formulation (Meier et al. 2021). For a
mutation wt->mut at position p:

    score = log P(mut at p | sequence) - log P(wt at p | sequence)

summed over a variant's mutations. Crucially this needs only ~L forward
passes per BACKBONE, not one per variant: the model is queried about the
four wild-type sequences, producing an (L x 20) log-probability matrix
each, and all 141,150 variants are then scored by table lookup.

    masked-marginals (default) — mask position p, then read its logits.
                                 L passes per backbone; the better scorer.
    wt-marginals               — one unmasked pass per backbone; ~L times
                                 faster, slightly weaker.

OFFLINE BY DESIGN. Nothing is downloaded. Point --checkpoint at a local
ESM-2 .pt file. See the note at the bottom of this file for what to fetch.

Run:  python esm_scores.py --checkpoint ../data/esm2_t12_35M_UR50D.pt
      python esm_scores.py --dry-run          # validate everything but the model

Output: output/esm_logprobs.npz     per-backbone (L x 20) log-prob matrices
        output/features/X_esm.npy   per-variant features, aligned to meta.csv
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"

MUT_RE = re.compile(r"([A-Z])(\d+)([A-Z])")
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
FEATURE_NAMES = ["esm_sum", "esm_mean", "esm_min", "esm_max"]


# --------------------------------------------------------------------------
# model scoring (needs torch + fair-esm + a local checkpoint)
# --------------------------------------------------------------------------

def compute_logprob_matrices_hf(wt_sequences: dict, hf_model: str, mode: str,
                                batch_size: int, cache: Path,
                                time_budget: float = 480.0) -> dict | None:
    """Same (L, 20) log-probability matrices, via HuggingFace transformers.

    Added when huggingface.co turned out to be reachable from the cloud
    container while dl.fbaipublicfiles.com (fair-esm's host) was still 403.
    The formulation, the token offset and the downstream feature code are
    unchanged — only the weights source differs.

    Resume-safe: each backbone's matrix is cached as it completes, so this
    can be re-run across several calls for the larger checkpoints.
    """
    import time
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    cache.mkdir(parents=True, exist_ok=True)
    tag = hf_model.split("/")[-1] + "_" + mode

    pending = [b for b in wt_sequences
               if not (cache / f"{tag}_{b}.npy").exists()]
    if pending:
        print(f"loading {hf_model} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(hf_model)
        model = AutoModelForMaskedLM.from_pretrained(hf_model)
        model.eval()
        torch.set_grad_enabled(False)
        aa_idx = [tok.convert_tokens_to_ids(a) for a in AMINO_ACIDS]
        assert all(i is not None and i >= 0 for i in aa_idx), "alphabet mismatch"

    started = time.time()
    for backbone in pending:
        seq = wt_sequences[backbone]
        L = len(seq)
        enc = tok(seq, return_tensors="pt")
        tokens = enc["input_ids"]
        # the residue at 0-based index i sits at token i+1 (BOS at 0); assert
        # it rather than assume it, since tokenisers differ
        assert tokens.shape[1] == L + 2, f"unexpected tokenisation: {tokens.shape}"
        for i, ch in enumerate(seq):
            assert tokens[0, i + 1].item() == tok.convert_tokens_to_ids(ch), \
                f"token offset wrong at {i}"

        out = np.zeros((L, 20), dtype=np.float32)
        if mode == "wt-marginals":
            lp = torch.log_softmax(model(tokens).logits[0], dim=-1)
            for i in range(L):
                out[i] = lp[i + 1, aa_idx].numpy()
        else:
            for start in range(0, L, batch_size):
                if time.time() - started > time_budget:
                    print(f"\n  time budget reached during {backbone} "
                          f"— re-run to continue")
                    return None
                positions = list(range(start, min(start + batch_size, L)))
                batch = tokens.repeat(len(positions), 1).clone()
                for row, pos in enumerate(positions):
                    batch[row, pos + 1] = tok.mask_token_id
                lp = torch.log_softmax(model(batch).logits, dim=-1)
                for row, pos in enumerate(positions):
                    out[pos] = lp[row, pos + 1, aa_idx].numpy()
                print(f"  {backbone}: {min(start + batch_size, L)}/{L} "
                      f"({time.time() - started:.0f}s)", end="\r", flush=True)

        np.save(cache / f"{tag}_{backbone}.npy", out)
        print(f"  {backbone}: {L} positions scored                    ")

    return {b: np.load(cache / f"{tag}_{b}.npy") for b in wt_sequences}


def compute_logprob_matrices(wt_sequences: dict, checkpoint: Path,
                             mode: str, batch_size: int) -> dict:
    """Return {backbone: (L, 20) array of log P(aa at position)}."""
    import torch
    import esm

    print(f"loading {checkpoint.name} ...", flush=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(checkpoint))
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    aa_idx = [alphabet.get_idx(a) for a in AMINO_ACIDS]

    matrices = {}
    for backbone, seq in wt_sequences.items():
        L = len(seq)
        _, _, tokens = batch_converter([(backbone, seq)])
        # token 0 is BOS, so residue i (0-based) sits at token i+1
        out = np.zeros((L, 20), dtype=np.float32)

        with torch.no_grad():
            if mode == "wt-marginals":
                logits = model(tokens)["logits"]
                lp = torch.log_softmax(logits[0], dim=-1)
                for i in range(L):
                    out[i] = lp[i + 1, aa_idx].numpy()
            else:
                for start in range(0, L, batch_size):
                    positions = list(range(start, min(start + batch_size, L)))
                    batch = tokens.repeat(len(positions), 1).clone()
                    for row, pos in enumerate(positions):
                        batch[row, pos + 1] = alphabet.mask_idx
                    logits = model(batch)["logits"]
                    lp = torch.log_softmax(logits, dim=-1)
                    for row, pos in enumerate(positions):
                        out[pos] = lp[row, pos + 1, aa_idx].numpy()
                    print(f"  {backbone}: {min(start + batch_size, L)}/{L}",
                          end="\r", flush=True)

        matrices[backbone] = out
        print(f"  {backbone}: {L} positions scored           ")
    return matrices


# --------------------------------------------------------------------------
# turning the matrices into per-variant features (no torch needed)
# --------------------------------------------------------------------------

def build_esm_features(meta: pd.DataFrame, matrices: dict,
                       wt_sequences: dict) -> np.ndarray:
    """Per-variant [sum, mean, min, max] of log-odds over its mutations."""
    aa_pos = {a: i for i, a in enumerate(AMINO_ACIDS)}
    X = np.zeros((len(meta), len(FEATURE_NAMES)), dtype=np.float32)
    skipped = 0

    for row, (backbone, name) in enumerate(
            zip(meta["Backbone"], meta["Variant name"])):
        matrix = matrices.get(backbone)
        if matrix is None:
            continue
        scores = []
        for wt, pos_s, mut in MUT_RE.findall(str(name)):
            p = int(pos_s) - 1
            if not (0 <= p < matrix.shape[0]) or wt not in aa_pos or mut not in aa_pos:
                skipped += 1
                continue
            # sanity: the stated wild-type residue should match the WT sequence
            if wt_sequences[backbone][p] != wt:
                skipped += 1
                continue
            scores.append(matrix[p, aa_pos[mut]] - matrix[p, aa_pos[wt]])
        if scores:
            s = np.array(scores, dtype=np.float32)
            X[row] = [s.sum(), s.mean(), s.min(), s.max()]

    if skipped:
        print(f"note: {skipped} mutations skipped (position or residue mismatch)")
    return X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        help="local ESM-2 .pt file; nothing is downloaded")
    parser.add_argument("--hf-model",
                        help="HuggingFace ESM-2 id, e.g. facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--suffix", default="",
                        help="suffix for the output filenames, to keep several "
                             "checkpoints' features side by side")
    parser.add_argument("--mode", choices=["masked-marginals", "wt-marginals"],
                        default="masked-marginals")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true",
                        help="exercise everything except the model, using a "
                             "random matrix, to validate wiring offline")
    args = parser.parse_args()

    wt_sequences = json.loads((OUT_DIR / "backbone_wt.json").read_text())
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    print(f"backbones: {', '.join(f'{b}({len(s)}aa)' for b, s in wt_sequences.items())}")
    print(f"variants to score: {len(meta)}")

    if args.dry_run:
        rng = np.random.default_rng(0)
        matrices = {b: rng.normal(-3, 1, size=(len(s), 20)).astype(np.float32)
                    for b, s in wt_sequences.items()}
        print("\n[dry run] using random log-probabilities — feature VALUES are "
              "meaningless, this only checks the plumbing")
    elif args.hf_model:
        matrices = compute_logprob_matrices_hf(
            wt_sequences, args.hf_model, args.mode, args.batch_size,
            OUT_DIR / "esm_cache")
        if matrices is None:
            return
        np.savez_compressed(OUT_DIR / f"esm_logprobs{args.suffix}.npz", **matrices)
    else:
        if not args.checkpoint or not args.checkpoint.exists():
            raise SystemExit(
                "no checkpoint. Download an ESM-2 .pt into the project folder "
                "and pass --checkpoint, or use --dry-run.")
        matrices = compute_logprob_matrices(
            wt_sequences, args.checkpoint, args.mode, args.batch_size)
        np.savez_compressed(OUT_DIR / "esm_logprobs.npz", **matrices)

    X = build_esm_features(meta, matrices, wt_sequences)
    nonzero = int((X != 0).any(axis=1).sum())
    print(f"\nesm features: {X.shape}, {nonzero} rows scored "
          f"({nonzero / len(meta):.1%})")
    print(pd.DataFrame(X, columns=FEATURE_NAMES).describe().round(3).to_string())

    if not args.dry_run:
        np.save(FEAT_DIR / f"X_esm{args.suffix}.npy", X)
        (FEAT_DIR / "esm_feature_names.txt").write_text("\n".join(FEATURE_NAMES))
        print(f"\nwrote {FEAT_DIR / f'X_esm{args.suffix}.npy'}")
    else:
        print("\n[dry run] nothing written")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# WHAT TO DOWNLOAD (this sandbox has no internet; put files in the project
# folder and they become visible here)
#
#   1. ESM-2 checkpoint — either is fine, start small to prove the pipeline:
#        esm2_t12_35M_UR50D.pt    ~150 MB   good first test
#        esm2_t33_650M_UR50D.pt   ~2.5 GB   standard for variant effects
#      from https://dl.fbaipublicfiles.com/fair-esm/models/<name>
#      Note: fair-esm also wants the matching *-contact-regression.pt for
#      some models; if load fails asking for it, fetch the same name with
#      "-contact-regression.pt" from that directory.
#
#   2. PyTorch CPU wheel for LINUX (not Windows — that is this sandbox's
#      platform), Python 3.10, e.g.
#        torch-2.*.*+cpu-cp310-cp310-linux_x86_64.whl
#      from https://download.pytorch.org/whl/cpu/torch/
#      Install offline with:  pip install --break-system-packages <wheel>
#      Its small dependencies come from PyPI, which IS reachable here.
# ---------------------------------------------------------------------------
