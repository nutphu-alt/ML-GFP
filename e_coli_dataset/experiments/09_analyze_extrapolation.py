"""
GFP variants dataset: step 4d — does the MLP's advantage survive real extrapolation?

The headline position-holdout numbers are dominated by multi-mutant test
rows that carry only ONE mutation at a held-out position; the rest of
their mutations sit at positions the model trained on. Step 3 showed the
additive models collapse to the mean-predictor floor once that crutch is
removed. This script asks the same question of the full-data MLP.

Models compared on the position-holdout test set:
  mean_by_backbone  the floor (from step 3 predictions)
  ridge_combined    best additive model (from step 3 predictions)
  xgboost           refit here; the tree model, for the interaction story
  mlp_full          best-epoch weights restored from the step-4c checkpoint

Run with: python analyze_extrapolation.py
Output: output/extrapolation_breakdown_mlp.csv, output/extrapolation_mlp.png
"""

import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"

MUT_RE = re.compile(r"[A-Z](\d+)[A-Z]")
EXPECTED_MLP_RHO = 0.8228     # reported by train_mlp_full.py; used as a self-check


def main() -> None:
    X_sparse = sparse.load_npz(FEAT_DIR / "X_sparse.npz")
    X_dense = np.load(FEAT_DIR / "X_dense.npy")
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")

    split = meta["split_position_holdout"].to_numpy()
    tr, te = (split == "train"), (split == "test")

    # rebuild the feature matrix exactly as train_mlp_full.py did, or the
    # restored weights would be applied to differently-scaled inputs
    scaler = StandardScaler().fit(X_dense[tr])
    X_dense_s = scaler.transform(X_dense).astype(np.float32)
    X = sparse.hstack([X_sparse, sparse.csr_matrix(X_dense_s)]).tocsr()
    y_te = y[te]

    preds = {}

    # step-3 predictions (floor + best additive model)
    step3 = np.load(OUT_DIR / "preds_position.npz")
    assert np.allclose(step3["y_true"], y_te), "step-3 test rows no longer align"
    preds["mean_by_backbone"] = step3["mean_by_backbone"]
    preds["ridge_combined"] = step3["ridge_combined"]

    # xgboost, refit with the config the step-4 search selected
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=0)
    model.fit(X[tr], y[tr])
    preds["xgboost"] = model.predict(X[te])

    # full-data MLP: restore the BEST epoch's weights, not the last epoch's.
    # 07_train_mlp_full.py names its checkpoint after --alpha, and its docstring
    # says to use 0.001 for the position split -- but the flag DEFAULTS to 1e-4,
    # so a run that followed the plain run order leaves alpha0.0001 instead.
    # Prefer the documented checkpoint, fall back to whatever exists, and say
    # which one was used rather than dying with a bare FileNotFoundError.
    ckpt_dir = OUT_DIR / "mlp_checkpoints"
    candidates = [ckpt_dir / "mlp_position_alpha0.001.joblib",
                  *sorted(ckpt_dir.glob("mlp_position_alpha*.joblib"))]
    ckpt_path = next((p for p in candidates if p.exists()), None)
    if ckpt_path is None:
        raise SystemExit(
            f"no position-split MLP checkpoint in {ckpt_dir}.\n"
            "Run:  python 07_train_mlp_full.py --split position --alpha 0.001")
    if ckpt_path.name != "mlp_position_alpha0.001.joblib":
        print(f"note: using {ckpt_path.name}; the documented run for this step is "
              "--split position --alpha 0.001")
    ckpt = joblib.load(ckpt_path)
    mlp = ckpt["model"]
    mlp.coefs_, mlp.intercepts_ = ckpt["best_weights"]
    preds["mlp_full"] = mlp.predict(X[te])

    rho = spearmanr(y_te, preds["mlp_full"]).statistic
    # EXPECTED_MLP_RHO is a reference value from the original session, not a
    # correctness target: the MLP early-stops, so a rerun with a different
    # epoch count legitimately lands a little away from it. Only a LARGE gap
    # means the reconstruction is actually wrong.
    gap = abs(rho - EXPECTED_MLP_RHO)
    if gap < 0.002:
        note = "matches the reference run"
    elif gap < 0.02:
        note = (f"differs from the reference run by {gap:.4f} — expected if the "
                "MLP stopped at a different epoch, not a fault")
    else:
        note = (f"differs from the reference run by {gap:.4f} — large enough to "
                "check that the right checkpoint was loaded")
    print(f"MLP overall test rho = {rho:.4f} "
          f"(reference {EXPECTED_MLP_RHO:.4f}; {note})\n")

    # ---- how much of each test row is genuinely unseen ---------------------
    positions = meta["Variant name"].apply(
        lambda n: [int(p) for p in MUT_RE.findall(str(n))])
    seen = {
        b: set().union(*g) if len(g) else set()
        for b, g in positions[tr].groupby(meta.loc[tr, "Backbone"])
    }
    test_meta = meta[te].copy()
    test_meta["positions"] = positions[te].values
    test_meta["n_mut"] = test_meta["positions"].apply(len)
    test_meta["n_novel"] = [
        sum(p not in seen[b] for p in ps)
        for b, ps in zip(test_meta["Backbone"], test_meta["positions"])
    ]

    subsets = {
        "all test rows": np.ones(len(test_meta), bool),
        "1 novel + others seen": ((test_meta["n_novel"] == 1)
                                  & (test_meta["n_mut"] > 1)).to_numpy(),
        "every mutation novel": (test_meta["n_novel"]
                                 == test_meta["n_mut"]).to_numpy(),
        "single mutant (pure extrapolation)": (test_meta["n_mut"] == 1).to_numpy(),
    }

    rows = []
    for label, mask in subsets.items():
        if mask.sum() > 10:
            row = {"subset": label, "n": int(mask.sum())}
            for name, p in preds.items():
                row[name] = spearmanr(y_te[mask], p[mask]).statistic
            rows.append(row)
    table = pd.DataFrame(rows).set_index("subset")
    table.to_csv(OUT_DIR / "extrapolation_breakdown_mlp.csv")
    print(table.round(4).to_string())

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    models = list(preds)
    x = np.arange(len(table))
    width = 0.8 / len(models)
    for i, name in enumerate(models):
        ax.bar(x + i * width - 0.4 + width / 2, table[name], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" (", "\n(") for s in table.index], fontsize=9)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Position-holdout test: performance by how much of the variant is unseen")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "extrapolation_mlp.png", dpi=130)
    print("\nwrote extrapolation_breakdown_mlp.csv and extrapolation_mlp.png")


if __name__ == "__main__":
    main()
