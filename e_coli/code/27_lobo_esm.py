"""
GFP variants dataset: step 11d — ESM embeddings on an entirely new protein.

Step 11b showed ESM-2 embeddings lift never-assayed-position performance from
rho 0.479 to 0.519. The other headline benchmark is leave-one-backbone-out:
train on three GFPs, predict the fourth. Step 9d put dense+evo at mean 0.524.

Fair by construction, the same argument step 9d made: for a held-out backbone
the evo features come from the OTHER three wild-types, and the ESM embeddings
come from the held-out backbone's own wild-type sequence — which you know if
you are engineering a new fluorescent protein. No brightness measurement from
the held-out protein is used by either block.

This is also the sharpest test of the ppluGFP2 question. ppluGFP2 is the
phylogenetic outlier (18-25% identity) that the evo features actively HURT,
and step 9 predicted a language model would not have that blind spot because
it has seen copepod fluorescent proteins the other three backbones know
nothing about.

Run with: python lobo_esm.py
Output: output/lobo_esm.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import importlib, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
_evo_benchmark = importlib.import_module("18_evo_benchmark")
XGB_CFG = _evo_benchmark.XGB_CFG

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "output"
FEAT_DIR = OUT_DIR / "features"


def main():
    meta = pd.read_csv(FEAT_DIR / "meta.csv")
    y = np.load(FEAT_DIR / "y.npy")
    Xd = np.load(FEAT_DIR / "X_dense.npy")
    Xe = np.load(FEAT_DIR / "X_evo.npy")
    Xb = np.load(FEAT_DIR / "X_esmemb_650M.npy")
    Xm = np.load(FEAT_DIR / "X_esm_650M.npy")

    blocks = {
        "dense (66)": Xd,
        "dense+evo (77)": np.hstack([Xd, Xe]),
        "dense+evo+esm (81)": np.hstack([Xd, Xe, Xm]),
        "dense+evo+emb (109)": np.hstack([Xd, Xe, Xb]),
    }
    bb = meta["Backbone"].to_numpy()
    backbones = sorted(set(bb))

    rows = []
    for label, X_all in blocks.items():
        scores = {}
        for held in backbones:
            train, test = bb != held, bb == held
            sc = StandardScaler().fit(X_all[train])
            X = sc.transform(X_all).astype(np.float32)
            t0 = time.time()
            m = xgb.XGBRegressor(tree_method="hist", n_jobs=-1, random_state=0,
                                 **XGB_CFG)
            m.fit(X[train], y[train])
            r = spearmanr(m.predict(X[test]), y[test]).statistic
            scores[held] = round(float(r), 4)
            print(f"  {label:22s} held out {held:9s} rho {r:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        scores["mean"] = round(float(np.mean([scores[b] for b in backbones])), 4)
        rows.append({"features": label, **scores})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "lobo_esm.csv", index=False)
    print("\n" + df.to_string(index=False))

    base = df[df["features"] == "dense+evo (77)"].iloc[0]
    best = df.loc[df["mean"].idxmax()]
    print(f"\nbest: {best['features']} at mean rho {best['mean']} "
          f"(step-9 winner dense+evo was {base['mean']})")
    print("ppluGFP2, the phylogenetic outlier: " + ", ".join(
        f"{r['features'].split(' ')[0]} {r['ppluGFP2']}" for _, r in df.iterrows()))


if __name__ == "__main__":
    main()
