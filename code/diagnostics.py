"""Diagnostics of the baseline: split-seed distribution, nuisance fit quality and sensitivity to
the learner's hyper-parameters.

* seeds: the baseline FE-DML estimate and the three tercile contrasts for 20 seeds of the
  sample splits (20 repetitions each);
* nuisance fit: out-of-fold R^2 (overall and within country) of the outcome and treatment
  nuisance functions, five folds by country, for LightGBM, a random forest and a linear
  correlated-random-effects regression with year dummies;
* hyper-parameters: FE-DML and contrasts for a grid of LightGBM settings.

Output: results/diagnostics.json.  Run as ``python code/diagnostics.py <n_jobs>``.
"""
import copy
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from empirical import BASE, N_ROB, STATIC, fit_dose, rf, tercile_masks  # noqa: E402
from panel_dose import mundlak_means, within_two_way  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MODS = [("A", "hc_l1"), ("B", "lp_init"), ("C", "ysince_l1")]


def summary(m, df):
    lo, _, hi = tercile_masks(m, df)
    d, se = m.group_contrast(hi, lo)
    c = copy.copy(m)
    c._raw = m._raw
    c.rebasis(0, 0, penalty=False)
    f, fse = c.average_effect()
    return {"fedml": f, "fedml_se": fse, "diff": d, "diff_se": se, "lambda": m.lambda_}


def one_seed(df, P, z, seed):
    d = df.dropna(subset=[z]).reset_index(drop=True)
    m = fit_dose(d, z=z, seed=seed)
    return {"panel": P, "seed": seed, **summary(m, d)}


def lgbm_factory(lr, leaves, trees=400):
    return lambda: lgb.LGBMRegressor(n_estimators=trees, learning_rate=lr, num_leaves=leaves,
                                     min_child_samples=20, subsample=0.8, subsample_freq=1,
                                     colsample_bytree=0.8, verbose=-1, n_jobs=1)


def one_hyper(df, P, z, lr, leaves, trees):
    d = df.dropna(subset=[z]).reset_index(drop=True)
    m = fit_dose(d, z=z, learner=lgbm_factory(lr, leaves, trees), n_rep=N_ROB)
    return {"panel": P, "lr": lr, "leaves": leaves, "trees": trees, **summary(m, d)}


def nuisance_fit(df):
    """Out-of-fold R^2 of E[Y|F] and E[D|F] with the baseline feature set."""
    controls = BASE + ["hc_l1"] if "hc_l1" not in BASE else BASE
    extra = [c for c in controls if c not in STATIC] + ["internet_l1"]
    F = pd.concat([df[controls], mundlak_means(df, "iso", extra)], axis=1)
    F["_t"] = df["year"]
    Fy = pd.concat([F.drop(columns="_t"), pd.get_dummies(df["year"], prefix="y").astype(float)],
                   axis=1)
    out = {}
    learners = {"LightGBM": (lgbm_factory(0.03, 15), F), "Random forest": (rf, F),
                "Linear (CRE, year dummies)": (LinearRegression, Fy)}
    for name, (fac, X) in learners.items():
        for tgt, col in [("outcome", "dlp"), ("treatment", "internet_l1")]:
            y = df[col].values
            pred = np.empty(len(y))
            for tr, te in GroupKFold(5).split(X, groups=df["iso"]):
                pred[te] = fac().fit(X.iloc[tr].values, y[tr]).predict(X.iloc[te].values)
            r = y - pred
            r2 = 1 - np.var(r) / np.var(y)
            yw = within_two_way(y, df["iso"], df["year"])
            rw = within_two_way(r, df["iso"], df["year"])
            out[f"{name}|{tgt}"] = {"r2": float(r2), "r2_within": float(1 - np.var(rw) / np.var(yw))}
    return out


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    out = {}
    seeds = Parallel(n_jobs=N_JOBS, verbose=2)(delayed(one_seed)(df, P, z, s)
                                               for P, z in MODS for s in range(100, 120))
    out["seeds"] = seeds
    s = pd.DataFrame(seeds)
    a = s[s["panel"] == "A"]
    out["seed_summary"] = {
        "fedml_mean": float(a["fedml"].mean()), "fedml_sd": float(a["fedml"].std()),
        "fedml_min": float(a["fedml"].min()), "fedml_max": float(a["fedml"].max()),
        "share_ci_excludes_zero": float(np.mean(a["fedml"] + 1.96 * a["fedml_se"] < 0)),
        "upper_median": float(np.median(a["fedml"] + 1.96 * a["fedml_se"])),
        **{f"diff_{P}_{k}": float(getattr(s[s["panel"] == P]["diff"], k)())
           for P in "ABC" for k in ["mean", "std", "min", "max"]}}
    print("seeds done", flush=True)
    grid = [(0.03, 15, 400), (0.01, 15, 1200), (0.1, 15, 150), (0.03, 7, 400), (0.03, 31, 400)]
    out["hyper"] = Parallel(n_jobs=N_JOBS)(delayed(one_hyper)(df, P, z, *g)
                                           for P, z in MODS[:2] for g in grid)
    print("hyper done", flush=True)
    out["nuisance_fit"] = nuisance_fit(df)
    json.dump(out, open(RES / "diagnostics.json", "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ["seed_summary", "nuisance_fit"]}, indent=1))


if __name__ == "__main__":
    main()
