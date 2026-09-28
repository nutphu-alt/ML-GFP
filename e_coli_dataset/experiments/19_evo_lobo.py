"""
GFP variants dataset: step 9d — do evolutionary features help a NEW protein?

Step 9c showed the cross-homolog features add +0.044 Spearman at unseen
positions within a known backbone. This asks the harder and more useful
question: do they help when the whole protein is new? That is the setting
someone actually engineering a novel FP is in.

Same leave-one-backbone-out protocol as step 7, same XGBoost config, only
the feature block changes (dense 66 vs dense+evo 77).

One design point worth stating, because it decides whether the comparison
is fair: for a held-out backbone the evo features are computed from the
OTHER three wild-type sequences. That is legitimate — if you are engineering
a new fluorescent protein you know its wild-type sequence, and aligning it
against known relatives costs nothing. No brightness measurement from the
held-out protein is used anywhere.

Run with: python evo_lobo.py
Output: output/evo_lobo.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"

XGB_CFG = dict(n_estimators=400, max_depth=8, learning_rate=0.1,
               subsample=0.8, colsample_bytree=0.8)
TIME_BUDGET = 150


def main() -> None:
    X_dense = np.load(FEAT_DIR / "X_dense.npy")
    X_evo = np.load(FEAT_DIR / "X_evo.npy")
    y = np.load(FEAT_DIR / "y.npy")
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    backbones = sorted(meta["Backbone"].unique())

    results_path = OUT_DIR / "evo_lobo.csv"
    done = set()
    if results_path.exists():
        prev = pd.read_csv(results_path)
        done = set(zip(prev["held_out"], prev["features"]))

    blocks = {"dense": X_dense, "dense_evo": np.hstack([X_dense, X_evo])}

    started = time.time()
    for held in backbones:
        test = (meta["Backbone"] == held).to_numpy()
        train = ~test
        for name, block in blocks.items():
            if (held, name) in done:
                continue
            if time.time() - started > TIME_BUDGET:
                print(f"time budget reached — re-run to continue "
                      f"(next: {held}/{name})")
                return
            scaler = StandardScaler().fit(block[train])
            X = scaler.transform(block).astype(np.float32)
            model = xgb.XGBRegressor(tree_method="hist", n_jobs=-1,
                                     random_state=0, **XGB_CFG)
            model.fit(X[train], y[train])
            rho = spearmanr(y[test], model.predict(X[test])).statistic
            pd.DataFrame([{"held_out": held, "features": name,
                           "n_test": int(test.sum()), "spearman": rho}]).to_csv(
                results_path, mode="a", header=not results_path.exists(),
                index=False)
            print(f"{held:9s} {name:10s} rho={rho:+.4f}", flush=True)

    df = pd.read_csv(results_path)
    pivot = df.pivot_table(index="features", columns="held_out", values="spearman")
    pivot["mean"] = pivot.mean(axis=1)
    print("\n=== leave-one-backbone-out, XGBoost, Spearman ===")
    print(pivot.round(4).to_string())
    delta = pivot.loc["dense_evo"] - pivot.loc["dense"]
    print("\ndelta (dense+evo − dense):")
    print(delta.round(4).to_string())


if __name__ == "__main__":
    main()
