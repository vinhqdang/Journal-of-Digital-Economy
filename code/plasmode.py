"""Plasmode simulation built on the empirical panel (Franklin et al., 2014 style).

The actual controls, treatment, moderator and panel structure of the baseline sample are kept.
The outcome is regenerated as

    Y*_it = theta(Z_it) D_it + g_hat(F_it) + w_i e_it,

where g_hat is a boosted-tree fit of observed productivity growth on the nuisance features,
e_it are the corresponding residuals and w_i are Rademacher weights drawn per country, so the
residuals keep their size, heteroskedasticity and within-country dependence while any residual
association with D is randomised away.

For each of the three moderators of the application (schooling, initial productivity, years
since take-off) we consider three effect functions: no effect (size of the constancy test), a
linear gradient (size of the linearity test, power of the constancy test) and a smooth threshold
(power and bias under curvature).  The linear and threshold functions are scaled so that their
high-minus-low tercile contrast equals the minimum detectable contrast of the application,
2.8 times its standard error in the baseline.  The estimator is run exactly as in the application
(LightGBM, five country folds, five repetitions of the split, cross-validated penalty).
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from empirical import BASE, STATIC, lgbm, twfe_gradient  # noqa: E402
from panel_dose import PanelDOSE, mundlak_means  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODS = {"hc_l1": "Schooling", "lp_init": "Initial productivity", "ysince_l1": "Years since take-off"}


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
    return df


def shape_fun(shape, z):
    """Unscaled effect function of the standardised moderator."""
    s = (z - np.median(z)) / np.std(z)
    if shape == "null":
        return np.zeros_like(z)
    if shape == "linear":
        return s
    if shape == "threshold":
        return 1 / (1 + np.exp(-3.0 * s))
    raise ValueError(shape)


def theta(shape, z, target):
    """Effect function centred at zero mean with high-minus-low tercile contrast = target."""
    f = shape_fun(shape, z)
    if shape == "null":
        return f, lambda g: np.zeros_like(g)
    q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
    c = f[z > q2].mean() - f[z <= q1].mean()
    a, b = target / c, f.mean()
    zz, med, sd = z, np.median(z), np.std(z)

    def at(grid):
        s = (grid - med) / sd
        base = s if shape == "linear" else 1 / (1 + np.exp(-3.0 * s))
        return a * (base - b)
    return at(zz), at


def one(zname, shape, rep, df, target):
    rng = np.random.default_rng(10_000 + rep + 100_000 * list(MODS).index(zname))
    d = df.dropna(subset=[zname]).reset_index(drop=True).copy()
    iso = d["iso"].unique()
    w = dict(zip(iso, rng.choice([-1.0, 1.0], len(iso))))
    z = d[zname].values
    th, at = theta(shape, z, target)
    d["dlp"] = th * d["internet_l1"] + d["_g"] + d["iso"].map(w) * d["_e"]
    q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
    lo, hi = z <= q1, z > q2
    true_diff = th[hi].mean() - th[lo].mean()
    grid = np.quantile(z, np.linspace(0.05, 0.95, 17))
    tgrid = at(grid)
    ctrl = BASE + ([zname] if zname not in BASE else [])
    out = []
    m = PanelDOSE(lgbm, n_rep=5, z_in_controls=True, mundlak_exclude=STATIC, seed=rep)
    m.fit(d, "dlp", "internet_l1", ctrl, "iso", "year", z=zname)
    for name, pen in [("Panel-DOSE", True), ("Panel-DOSE (unpenalised)", False)]:
        m.refit_final(penalty=pen)
        e = m.effect(grid)
        diff, se = m.group_contrast(hi, lo)
        st = m.shape_tests()
        out.append({"moderator": zname, "shape": shape, "rep": rep, "method": name,
                    "irmse": float(np.sqrt(np.mean((e["est"] - tgrid) ** 2))),
                    "diff_err": diff - true_diff,
                    "diff_cover": float(abs(diff - true_diff) <= 1.96 * se),
                    "reject_const": float(st["const"][1] < 0.05),
                    "reject_lin": float(st["lin"][1] < 0.05),
                    "p_const": st["const"][1], "p_lin": st["lin"][1],
                    "reject_diff": float(abs(diff / se) > 1.96),
                    "cover": float(np.mean(np.abs(e["est"] - tgrid) <= 1.96 * e["se"])),
                    "ucover": float(np.all((tgrid >= e["ulo"]) & (tgrid <= e["uhi"]))),
                    "lambda": m.lambda_, "edf": m.edf_})
    if zname != "ysince_l1":
        for name, spline in [("TWFE-interaction", False), ("TWFE-spline", True)]:
            r = twfe_gradient(d, zname, spline=spline)
            out.append({"moderator": zname, "shape": shape, "rep": rep, "method": name,
                        "diff_err": r["diff"] - true_diff,
                        "diff_cover": float(abs(r["diff"] - true_diff) <= 1.96 * r["diff_se"]),
                        "reject_diff": float(abs(r["diff"] / r["diff_se"]) > 1.96)})
    return out


def main(reps=200, n_jobs=4):
    df = setup()
    summ = pd.read_csv(RES / "empirical_summary.csv")
    base = summ[summ["spec"] == "Baseline"].set_index("panel")
    targets = {"hc_l1": 2.8 * base.loc["A", "diff_se"], "lp_init": 2.8 * base.loc["B", "diff_se"],
               "ysince_l1": 2.8 * base.loc["C", "diff_se"]}
    tasks = []
    for zname in MODS:
        tasks += [(zname, "null", r) for r in range(reps)]
        tasks += [(zname, "linear", r) for r in range(reps // 2)]
        tasks += [(zname, "threshold", r) for r in range(reps // 2)]
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(zn, s, r, df, targets[zn])
                                             for zn, s, r in tasks)
    out = pd.DataFrame([x for y in res for x in y])
    out["target"] = out["moderator"].map(targets)
    out.to_csv(RES / "plasmode_raw.csv", index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200,
         int(sys.argv[2]) if len(sys.argv) > 2 else 4)
