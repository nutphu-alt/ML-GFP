"""
GFP variants dataset: step 7 — leave-one-backbone-out evaluation.

The hardest generalisation test in the project, and the last one pending
from step 2: train on three GFP backbones, predict the fourth. The test
protein has a different wild-type sequence, so nothing about its specific
positions was ever seen.

Two feature sets are compared deliberately:

  dense     the 66 transferable descriptors (BLOSUM62, hydropathy/volume/
            charge deltas, amino-acid composition, mutation count)
  combined  dense + the 6,559 (backbone, position, mutant AA) indicators

The sparse indicators are keyed on backbone, so for a held-out protein
every one of its columns is zero throughout training. Comparing the two
shows directly how much of each model's usual performance was riding on
position memorisation that cannot transfer.

Scored with Spearman ONLY. Fold-WT brightness is not comparable across
libraries — step 1 found the same protein reading 1.000 in one library and
0.245 in another — so RMSE and R² across backbones would be meaningless.

Run with: python lobo_eval.py
Output: output/lobo_results.csv, output/lobo.png
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"


def make_models():
    return {
        "ridge": lambda: Ridge(alpha=10.0, solver="lsqr"),
        "xgboost": lambda: xgb.XGBRegressor(
            n_estimators=400, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            n_jobs=-1, random_state=0),
        "mlp": lambda: MLPRegressor(
            hidden_layer_sizes=(256,), alpha=1e-3, max_iter=60,
            early_stopping=True, random_state=0),
    }


def main() -> None:
    X_sparse = sparse.load_npz(FEAT_DIR / "X_sparse.npz")
    X_dense = np.load(FEAT_DIR / "X_dense.npy")
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    backbones = sorted(meta["Backbone"].unique())

    # resume-safe: each result is appended as it finishes, and completed
    # combinations are skipped, so this can span several runs
    results_path = OUT_DIR / "lobo_results.csv"
    done = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        done = set(zip(prev["held_out"], prev["features"], prev["model"]))

    started = time.time()
    for held in backbones:
        test = (meta["Backbone"] == held).to_numpy()
        train = ~test

        scaler = StandardScaler().fit(X_dense[train])
        Xd = scaler.transform(X_dense).astype(np.float32)
        Xc = None

        for block in ["dense", "combined"]:
            for name, factory in make_models().items():
                # MLP is dense-only: on the 6,625-column block a single fit
                # runs past the wall-clock limit and cannot be checkpointed
                if block == "combined" and name == "mlp":
                    continue
                if (held, block, name) in done:
                    continue
                if time.time() - started > 100:
                    print(f"time budget reached — re-run to continue "
                          f"(next: {held}/{block}/{name})")
                    return

                if block == "combined" and Xc is None:
                    Xc = sparse.hstack([X_sparse, sparse.csr_matrix(Xd)]).tocsr()
                X = Xd if block == "dense" else Xc

                model = factory()
                model.fit(X[train], y[train])
                rho = spearmanr(y[test], model.predict(X[test])).statistic
                row = {"held_out": held, "features": block, "model": name,
                       "n_test": int(test.sum()), "spearman": rho}
                pd.DataFrame([row]).to_csv(
                    results_path, mode="a",
                    header=not results_path.exists(), index=False)
                print(f"{held:9s} {block:9s} {name:8s} rho={rho:+.4f}", flush=True)

    df = pd.read_csv(results_path)

    pivot = df.pivot_table(index=["features", "model"], columns="held_out",
                           values="spearman")
    pivot["mean"] = pivot.mean(axis=1)
    print("\n=== leave-one-backbone-out, Spearman ===")
    print(pivot.round(4).to_string())

    fig, ax = plt.subplots(figsize=(11, 5))
    combos = [(b, m) for b in ["dense", "combined"]
              for m in ["ridge", "xgboost", "mlp"]
              if not (b == "combined" and m == "mlp")]
    x = np.arange(len(backbones))
    width = 0.15
    for i, (block, model) in enumerate(combos):
        sub = df[(df.features == block) & (df.model == model)]
        vals = [sub[sub.held_out == b]["spearman"].iloc[0] for b in backbones]
        ax.bar(x + (i - (len(combos) - 1) / 2) * width, vals, width,
               label=f"{model} ({block})")
    ax.set_xticks(x)
    ax.set_xticklabels(backbones)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Spearman ρ on the held-out backbone")
    ax.set_title("Leave-one-backbone-out: predicting a protein never trained on")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "lobo.png", dpi=130)
    print("\nwrote lobo_results.csv and lobo.png")


if __name__ == "__main__":
    main()
