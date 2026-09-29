"""Plasmode simulation built on the empirical panel (Franklin et al., 2014 style).

The actual controls, treatment, moderator (initial log GDP per worker) and panel structure are
kept.  The outcome is regenerated as

    Y*_it = theta(Z_i) D_it + g_hat(F_it) + w_i e_it,

where g_hat is a boosted-tree fit of observed productivity growth on the nuisance features,
e_it are the corresponding residuals and w_i are Rademacher weights drawn per country, so the
residuals keep their size, heteroskedasticity and within-country dependence while any residual
association with D is randomised away.  Three effect functions: no effect (size), a linear
gradient whose high-minus-low tercile contrast equals the gradient reported for the original
specification (-9.5), and a threshold.
"""
import copy
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from empirical import BASE, STATIC, fe_ols, lgbm, twfe_gradient  # noqa: E402
from panel_dose import PanelDOSE, mundlak_means  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
Z = "lp_init"


def setup():
    df = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    feats = df[BASE].copy()
    extra = [c for c in BASE if c not in STATIC] + ["internet_l1"]
    feats = pd.concat([feats, mundlak_means(df, "iso", extra)], axis=1)
    feats["_t"] = df["year"]
    g = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                          min_child_samples=20, verbose=-1, n_jobs=1).fit(feats, df["dlp"])
    df["_g"] = g.predict(feats)
    df["_e"] = df["dlp"] - df["_g"]
    z = df[Z].values
    q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
    gap = z[z > q2].mean() - z[z <= q1].mean()
    return df, gap


def theta(shape, z, gap, zbar):
    if shape == "null":
        return np.zeros_like(z)
    if shape == "linear":
        return -9.5 / gap * (z - zbar)
    if shape == "threshold":
        return 6.0 - 12.0 / (1 + np.exp(-4.0 * (z - zbar)))
    raise ValueError(shape)


def one(shape, rep, df, gap):
    rng = np.random.default_rng(10_000 + rep)
    iso = df["iso"].unique()
    w = dict(zip(iso, rng.choice([-1.0, 1.0], len(iso))))
    z = df[Z].values
    zbar = z.mean()
    th = theta(shape, z, gap, zbar)
    d = df.copy()
    d["dlp"] = th * d["internet_l1"] + d["_g"] + d["iso"].map(w) * d["_e"]
    q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
    lo, hi = z <= q1, z > q2
    true_diff = th[hi].mean() - th[lo].mean()
    grid = np.quantile(z, np.linspace(0.05, 0.95, 17))
    tgrid = theta(shape, grid, gap, zbar)
    out = []
    m = PanelDOSE(lgbm, n_rep=2, z_in_controls=True, mundlak_exclude=STATIC, seed=rep)
    m.fit(d, "dlp", "internet_l1", BASE, "iso", "year", z=Z)
    for name, pen in [("Panel-DOSE", True), ("Panel-DOSE (unpenalised)", False)]:
        m.refit_final(penalty=pen)
        e = m.effect(grid)
        diff, se = m.group_contrast(hi, lo)
        out.append({"shape": shape, "rep": rep, "method": name,
                    "irmse": float(np.sqrt(np.mean((e["est"] - tgrid) ** 2))),
                    "diff_err": diff - true_diff, "diff_cover": float(abs(diff - true_diff) <= 1.96 * se),
                    "reject_const": float(m.shape_tests()["const"][1] < 0.05),
                    "reject_diff": float(abs(diff / se) > 1.96),
                    "cover": float(np.mean(np.abs(e["est"] - tgrid) <= 1.96 * e["se"])),
                    "ucover": float(np.all((tgrid >= e["ulo"]) & (tgrid <= e["uhi"]))),
                    "lambda": m.lambda_, "edf": m.edf_})
    for name, spline in [("TWFE-interaction", False), ("TWFE-spline", True)]:
        r = twfe_gradient(d, Z, spline=spline)
        out.append({"shape": shape, "rep": rep, "method": name, "diff_err": r["diff"] - true_diff,
                    "diff_cover": float(abs(r["diff"] - true_diff) <= 1.96 * r["diff_se"]),
                    "reject_diff": float(abs(r["diff"] / r["diff_se"]) > 1.96)})
    return out


def main(reps=100, n_jobs=3):
    df, gap = setup()
    tasks = [(s, r) for s in ["null", "linear"] for r in range(reps)]
    tasks += [("threshold", r) for r in range(reps // 2)]
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(s, r, df, gap) for s, r in tasks)
    pd.DataFrame([x for y in res for x in y]).to_csv(RES / "plasmode_raw.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100,
         int(sys.argv[2]) if len(sys.argv) > 2 else 3)
