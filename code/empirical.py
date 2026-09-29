"""Empirical application: internet diffusion and labour-productivity growth, 1996-2025.

Baseline (revised): controls are lagged or predetermined covariates that are not mechanically
related to the outcome (no lagged productivity level or growth), no other ICT measures and no
investment share, so that theta is the within-country association of internet diffusion with
subsequent productivity growth, conditional on schooling, government size, openness,
demography, urbanisation and initial productivity.  The development moderator is initial
(pre-sample) log GDP per worker.
"""
import copy
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_dose import PanelDOSE, bspline_basis, within_two_way  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PROC = ROOT / "data" / "processed"
RES.mkdir(exist_ok=True)

BASE = ["hc_l1", "csh_g_l1", "open_l1", "dpop_l1", "dep_l1", "urb_l1", "lp_init"]
ORIGINAL = ["lp_l1", "hc_l1", "csh_i_l1", "csh_g_l1", "open_l1", "dpop_l1", "dep_l1", "urb_l1",
            "dlp_l1", "mobile_l1", "broadband_l1"]
PWT_BASE = ["hc_l1", "csh_g_l1", "open_l1", "dpop_l1", "labsh_l1", "lp_init"]
STATIC = ("lp_init", "lays_mean", "spi_mean", "fixed90", "resource_share")
N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 1


def lgbm():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                             min_child_samples=20, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, n_jobs=1)


def rf():
    return RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                 n_jobs=1, random_state=0)


# ----------------------------------------------------------------------------- estimation
def fit_dose(df, y="dlp", d="internet_l1", z="hc_l1", controls=None, learner=lgbm, n_rep=5,
             mode="vc", **kw):
    controls = list(BASE if controls is None else controls)
    if mode == "vc" and z not in controls:
        controls.append(z)
    m = PanelDOSE(learner, mode=mode, n_rep=n_rep, z_in_controls=(mode == "vc"),
                  mundlak_exclude=STATIC, seed=kw.pop("seed", 11), **kw)
    m.fit(df.reset_index(drop=True), y, d, controls, "iso", "year",
          z=z if mode == "vc" else None)
    return m


def tercile_masks(m, df, by_year=False):
    z = m._Z
    if by_year:
        r = df.groupby("year")[m._zname].rank(pct=True).values
        return r <= 1 / 3, (r > 1 / 3) & (r <= 2 / 3), r > 2 / 3
    q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
    return z <= q1, (z > q1) & (z <= q2), z > q2


def summarise(m, name, panel, df, grid, by_year=False, weights=None):
    m._zname = getattr(m, "_zname", None)
    r = {"panel": panel, "spec": name, "n_obs": len(df), "n_countries": df["iso"].nunique()}
    lo_m, mid_m, hi_m = tercile_masks(m, df, by_year)
    z = m._Z
    p5 = np.quantile(z, 0.05)

    def block(suffix):
        out = {}
        out["avg_theta"], out["avg_theta_se"] = m.average_effect(weights)
        for lab, mk in [("low", lo_m), ("mid", mid_m), ("high", hi_m)]:
            out[f"gate_{lab}"], out[f"gate_{lab}_se"] = m.group_effect(mk)
        out["gate_low_x5"], out["gate_low_x5_se"] = m.group_effect(lo_m & (z > p5))
        out["diff"], out["diff_se"] = m.group_contrast(hi_m, lo_m)
        out["diff_x5"], out["diff_x5_se"] = m.group_contrast(hi_m, lo_m & (z > p5))
        out["blp"], out["blp_se"] = m.blp_slope()
        st = m.shape_tests()
        out["p_const"], out["p_lin"] = st["const"][1], st["lin"][1]
        out["lambda"], out["edf"] = m.lambda_, m.edf_
        return {k + suffix: v for k, v in out.items()}

    r.update(block(""))
    curve = m.effect(grid).to_dict(orient="list")
    m.refit_final(penalty=False)
    r.update(block("_u"))
    curve_u = m.effect(grid).to_dict(orient="list")
    m.refit_final(penalty=True)
    # headline average: constant-effect FE-DML on the same residuals
    c = copy.copy(m)
    c._raw = m._raw
    c.rebasis(0, 0, penalty=False)
    r["fedml"], r["fedml_se"] = c.average_effect()
    iv = m.identifying_variation({"low": lo_m, "mid": mid_m, "high": hi_m})
    if iv:
        r.update({f"idvar_{k}": v for k, v in iv.items()})
    for lab, mk in [("low", lo_m), ("mid", mid_m), ("high", hi_m)]:
        dsub = df.reset_index(drop=True)[mk]
        r[f"d_median_{lab}"] = float(np.median(m._D[mk]))
        r[f"d_within_sd_{lab}"] = float(dsub.groupby("iso")[m._dname].std().mean())
        r[f"n_countries_{lab}"] = int(dsub["iso"].nunique())
    for p in r:
        if isinstance(r[p], (np.floating,)):
            r[p] = float(r[p])
    return r, {"pen": curve, "unpen": curve_u}


def run_spec(spec):
    panel, name, data, kw = spec["panel"], spec["name"], spec["data"], dict(spec["kw"])
    z = kw.get("z", "hc_l1")
    grid = np.quantile(data[z], np.linspace(0.01, 0.99, 33))
    by_year = kw.pop("by_year", False)
    weights = kw.pop("weights", None)
    data = data.dropna(subset=[z, kw.get("y", "dlp"), kw.get("d", "internet_l1")])
    grid = np.quantile(data[z], np.linspace(0.01, 0.99, 33))
    m = fit_dose(data, **kw)
    m._zname, m._dname = z, kw.get("d", "internet_l1")
    w = data[weights].values if weights else None
    r, cv = summarise(m, name, panel, data.reset_index(drop=True), grid, by_year, w)
    print(panel, name, "done", flush=True)
    return r, cv


# ----------------------------------------------------------------------------- linear tools
def fe_ols(df, X, y, cluster="iso"):
    Xw = within_two_way(X, df["iso"], df["year"])
    yw = within_two_way(y, df["iso"], df["year"])
    b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    e = yw - Xw @ b
    u = pd.factorize(df[cluster])[0]
    S = np.zeros((u.max() + 1, Xw.shape[1]))
    np.add.at(S, u, Xw * e[:, None])
    A = np.linalg.pinv(Xw.T @ Xw)
    G, n, k = u.max() + 1, len(yw), Xw.shape[1]
    return b, G / (G - 1) * (n - 1) / max(n - k, 1) * A @ S.T @ S @ A


def twfe_gradient(df, z="lp_init", controls=BASE, spline=False):
    """TWFE with D x Z interaction (linear) or spline controls + spline effect; tercile contrast."""
    d, zz = df["internet_l1"].values, df[z].values
    q1, q2 = np.quantile(zz, [1 / 3, 2 / 3])
    if spline:
        kn = np.quantile(zz, [0.25, 0.5, 0.75])
        basis = lambda s: bspline_basis(s, kn, zz.min(), zz.max())  # noqa: E731
        ctrl = np.hstack([bspline_basis(df[c].values, np.quantile(df[c], [0.25, .5, .75]),
                                        df[c].min(), df[c].max())[:, 1:]
                          for c in controls if df[c].nunique() > 10 and c not in STATIC]
                         + [df[[c for c in controls if c in STATIC]].values])
    else:
        basis = lambda s: np.c_[np.ones(len(s)), s]  # noqa: E731
        ctrl = df[controls].values
    Bz = basis(zz)
    b, V = fe_ols(df, np.c_[d[:, None] * Bz, ctrl], df["dlp"].values)
    k = Bz.shape[1]
    b, V = b[:k], V[:k, :k]
    l = Bz[zz > q2].mean(0) - Bz[zz <= q1].mean(0)
    out = {"diff": float(l @ b), "diff_se": float(np.sqrt(l @ V @ l))}
    if not spline:
        out["slope"], out["slope_se"] = float(b[1]), float(np.sqrt(V[1, 1]))
    return out


def iv_check(df, controls=BASE):
    """Linear FE 2SLS: instrument = 1990 fixed-line density x leave-one-out global internet share.

    Controls, country and year effects are partialled out (Frisch-Waugh-Lovell); the
    just-identified 2SLS coefficient and its country-clustered standard error follow.
    """
    df = df.dropna(subset=["fixed90"]).reset_index(drop=True)
    tot = df.groupby("year")["internet_l1"].transform("sum")
    cnt = df.groupby("year")["internet_l1"].transform("count")
    glob = (tot - df["internet_l1"]) / (cnt - 1)
    zi = (df["fixed90"] * glob).values
    ctrl = within_two_way(df[[c for c in controls if c not in STATIC]].values, df["iso"],
                          df["year"])

    def res(v):
        vw = within_two_way(v, df["iso"], df["year"])
        b, *_ = np.linalg.lstsq(ctrl, vw, rcond=None)
        return vw - ctrl @ b

    y, d, z = res(df["dlp"].values), res(df["internet_l1"].values), res(zi)
    u = pd.factorize(df["iso"])[0]
    G = u.max() + 1

    def clus(num_resid, denom):
        S = np.bincount(u, num_resid, G)
        return float(np.sqrt(G / (G - 1) * np.sum(S ** 2)) / abs(denom))

    fs = np.sum(z * d) / np.sum(z * z)
    rf_ = np.sum(z * y) / np.sum(z * z)
    iv = np.sum(z * y) / np.sum(z * d)
    fs_se = clus(z * (d - fs * z), np.sum(z * z))
    return {"first_stage": float(fs), "first_stage_se": fs_se,
            "first_stage_F": float((fs / fs_se) ** 2), "reduced_form": float(rf_),
            "reduced_form_se": clus(z * (y - rf_ * z), np.sum(z * z)), "iv": float(iv),
            "iv_se": clus(z * (y - iv * d), np.sum(z * d)), "n_obs": len(df),
            "n_countries": int(df["iso"].nunique())}


# ----------------------------------------------------------------------------- dynamics
def five_year(df, controls=BASE):
    df = df.copy()
    df["period"] = ((df["year"] - 1996) // 5).clip(upper=5)
    first = df.sort_values("year").groupby(["iso", "period"]).first().reset_index()
    avg = df.groupby(["iso", "period"])["dlp"].mean().rename("dlp5").reset_index()
    n = df.groupby(["iso", "period"])["dlp"].size().rename("nyr").reset_index()
    p = first.drop(columns=["dlp"]).merge(avg, on=["iso", "period"]).merge(n, on=["iso", "period"])
    p = p[p["nyr"] >= 3].copy()
    p["year"] = p["period"]
    b, V = fe_ols(p, p[["internet_l1"] + controls].values, p["dlp5"].values)
    m = PanelDOSE(lgbm, n_rep=5, n_knots=0, degree=0, penalty=False, z_in_controls=True,
                  n_folds=5, mundlak_exclude=STATIC, seed=11)
    m.fit(p, "dlp5", "internet_l1", controls + ["hc_l1"] if "hc_l1" not in controls else controls,
          "iso", "year", z="hc_l1")
    a = m.average_effect()
    return {"twfe": float(b[0]), "twfe_se": float(np.sqrt(V[0, 0])), "fedml": a[0],
            "fedml_se": a[1], "n_obs": len(p), "n_countries": int(p["iso"].nunique())}


def long_difference(df, start=2000, end=2024, controls=BASE):
    """Cross-country long difference: annualised productivity growth start->end on the change in
    internet share, with start-year controls; OLS (HC1) and cross-fitted partially linear DML."""
    a = df[df["year"] == start].set_index("iso")
    b = df[df["year"] == end].set_index("iso")
    raw = pd.read_csv(ROOT / "data" / "raw" / "wdi" / "gdp_per_worker.csv")
    lp = np.log(raw.pivot(index="iso", columns="year", values="gdp_per_worker"))
    ids = a.index.intersection(b.index)
    y = 100 * (lp.loc[ids, end] - lp.loc[ids, start - 1]) / (end - start + 1)
    x = b.loc[ids, "internet_l1"] - a.loc[ids, "internet_l1"]
    W = a.loc[ids, controls]
    ok = y.notna() & x.notna() & W.notna().all(axis=1)
    y, x, W = y[ok].values, x[ok].values, W[ok].values
    X = np.c_[np.ones(len(y)), x, W]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    A = np.linalg.pinv(X.T @ X)
    Vh = len(y) / (len(y) - X.shape[1]) * A @ (X.T * e ** 2) @ X @ A
    ests, vs = [], []
    for s in range(5):
        ry, rx = np.empty(len(y)), np.empty(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=s).split(W):
            f = lambda: lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=4,  # noqa
                                          min_child_samples=8, verbose=-1, n_jobs=1)
            ry[te] = y[te] - f().fit(W[tr], y[tr]).predict(W[te])
            rx[te] = x[te] - f().fit(W[tr], x[tr]).predict(W[te])
        th = np.sum(rx * ry) / np.sum(rx ** 2)
        u = ry - th * rx
        ests.append(th)
        vs.append(np.sum(rx ** 2 * u ** 2) / np.sum(rx ** 2) ** 2)
    med = float(np.median(ests))
    v = float(np.median(np.array(vs) + (np.array(ests) - med) ** 2))
    return {"ols": float(beta[1]), "ols_se": float(np.sqrt(Vh[1, 1])), "dml": med,
            "dml_se": float(np.sqrt(v)), "n": int(len(y)), "mean_dx": float(np.mean(x))}


# ----------------------------------------------------------------------------- vintages
def vintage_decomposition(p10, p11):
    keys = ["iso", "year"]
    cols = ["dlp", "lp_init", "internet_l1"] + [c for c in PWT_BASE if c != "lp_init"]
    c = p10[keys + cols].merge(p11[keys + cols], on=keys, suffixes=("_10", "_11"))
    c = c.rename(columns={"internet_l1_10": "internet_l1"}).drop(columns=["internet_l1_11"])
    xs = [x for x in PWT_BASE if x != "lp_init"]
    combos = [("PWT 10.0 (all components)", "10", "10", "10"),
              ("Outcome from PWT 11.0", "11", "10", "10"),
              ("Moderator from PWT 11.0", "10", "11", "10"),
              ("Controls from PWT 11.0", "10", "10", "11"),
              ("PWT 11.0 (all components)", "11", "11", "11")]
    rows, infl = [], {}
    for name, vy, vz, vx in combos:
        m = fit_dose(c, y=f"dlp_{vy}", z=f"lp_init_{vz}", controls=[f"{x}_{vx}" for x in xs])
        z = m._Z
        q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
        lo, hi = z <= q1, z > q2
        diff, se = m.group_contrast(hi, lo)
        gl, gls = m.group_effect(lo)
        l = m._L_mean(hi) - m._L_mean(lo)
        phi, adj = m.influence(l)
        infl[name] = (phi, adj)
        ll = m._L_mean(lo)
        phil, _ = m.influence(ll)
        top = phil.abs().sort_values(ascending=False).head(5)
        rows.append({"combo": name, "diff": diff, "diff_se": se, "gate_low": gl,
                     "gate_low_se": gls, "lambda": m.lambda_,
                     "loco_top": "; ".join(f"{k} ({-phil[k]:+.2f})" for k in top.index)})
        print("vintage", name, "done", flush=True)
    a, b = infl["PWT 10.0 (all components)"], infl["PWT 11.0 (all components)"]
    dphi = a[0] - b[0].reindex(a[0].index)
    d_est = rows[0]["diff"] - rows[-1]["diff"]
    d_se = float(np.sqrt(a[1] * np.sum(dphi ** 2)))
    t10 = pd.qcut(c["lp_init_10"], 3, labels=False)
    t11 = pd.qcut(c["lp_init_11"], 3, labels=False)
    return {"rows": rows, "diff_of_diffs": d_est, "diff_of_diffs_se": d_se,
            "n_common": len(c), "n_countries": int(c["iso"].nunique()),
            "tercile_switch_share": float(np.mean(t10 != t11)),
            "countries_switching": int(c.loc[t10 != t11, "iso"].nunique())}


def dose_response(df):
    """Dose-response model f(D) in internet penetration (O'Sullivan penalty; R13 re-examination)."""
    m = PanelDOSE(lgbm, mode="dose", n_rep=5, mundlak_exclude=STATIC, seed=11)
    m.fit(df.reset_index(drop=True), "dlp", "internet_l1", BASE, "iso", "year")
    grid = np.quantile(df["internet_l1"], np.linspace(0.05, 0.95, 33))
    out = {"curve": m.effect(grid).to_dict(orient="list"), "avg": m.average_effect(),
           "p_linear_f": m.shape_tests()["const"][1], "lambda": m.lambda_, "edf": m.edf_}
    m.refit_final(penalty=False)
    out["curve_u"] = m.effect(grid).to_dict(orient="list")
    out["p_linear_f_u"] = m.shape_tests()["const"][1]
    return out


# ----------------------------------------------------------------------------- main
def main():
    df = pd.read_csv(PROC / "panel.csv")
    unt = pd.read_csv(PROC / "panel_untrimmed.csv")
    p10 = pd.read_csv(PROC / "panel_pwt.csv")
    p11 = pd.read_csv(PROC / "panel_pwt11.csv")
    out, curves, extra = [], {}, {}

    desc_cols = ["dlp", "internet_l1", "mobile_l1", "broadband_l1", "hc_l1", "lp_init",
                 "csh_i_l1", "csh_g_l1", "open_l1", "dpop_l1", "dep_l1", "urb_l1",
                 "ysince_l1", "dgdppc"]
    df[desc_cols].describe().T[["count", "mean", "std", "min", "max"]].to_csv(
        RES / "descriptives.csv")

    lo_q = df["lp_init"].quantile([0.01, 0.99])
    no_ict = [c for c in ORIGINAL if c not in ("mobile_l1", "broadband_l1")]
    # composite ICT index (0-1): first principal component of the three adoption shares
    ict = df[["internet_l1", "mobile_l1", "broadband_l1"]]
    zs = (ict - ict.mean()) / ict.std()
    w = np.linalg.eigh(np.cov(zs.T))[1][:, -1]
    pc = zs.values @ (w * np.sign(w.sum()))
    df["ict_index"] = (pc - pc.min()) / (pc.max() - pc.min())
    df_scr_off = pd.read_csv(PROC / "panel_unscreened.csv")
    s_pwt11 = df[df["iso"].isin(p11["iso"].unique()) & (df["year"] <= 2023)]
    hi_spi = df[df["spi_mean"] >= df.groupby("iso")["spi_mean"].first().median()]
    nores = df[df["resource_share"] < 50]
    wins = unt.copy()
    ql, qh = wins["dlp"].quantile([0.01, 0.99])
    wins["dlp"] = wins["dlp"].clip(ql, qh)

    def S(panel, name, data, **kw):
        return {"panel": panel, "name": name, "data": data, "kw": kw}

    specs = []
    for P, z in [("A", "hc_l1"), ("B", "lp_init"), ("C", "ysince_l1")]:
        specs.append(S(P, "Baseline", df, z=z))
    for P, z in [("A", "hc_l1"), ("B", "lp_init")]:
        specs += [
            S(P, "Original specification", df, z=z, controls=ORIGINAL + (["lp_init"] if z == "lp_init" else [])),
            S(P, "Adding the investment share", df, z=z, controls=BASE + ["csh_i_l1"]),
            S(P, "Adding mobile and broadband", df, z=z, controls=BASE + ["mobile_l1", "broadband_l1"]),
            S(P, "Treatment: composite ICT index", df, z=z, d="ict_index"),
            S(P, "Random forest nuisance", df, z=z, learner=rf, n_rep=3),
            S(P, "Unpenalised sieve", df, z=z, penalty=False, n_rep=3),
            S(P, "Three interior knots", df, z=z, n_knots=3, n_rep=3),
            S(P, "GCV penalty", df, z=z, select="gcv", n_rep=3),
            S(P, "Two-year lag of internet", df, z=z, d="internet_l2", n_rep=3),
            S(P, "Pre-COVID sample, 1996-2019", df[df["year"] <= 2019], z=z, n_rep=3),
            S(P, "Excluding 2020-2021", df[~df["year"].isin([2020, 2021])], z=z, n_rep=3),
            S(P, "Excluding 2025", df[df["year"] <= 2024], z=z, n_rep=3),
            S(P, "Untrimmed outcome", unt, z=z, n_rep=3),
            S(P, "Winsorised outcome", wins, z=z, n_rep=3),
            S(P, "Without data screening", df_scr_off, z=z, n_rep=3),
            S(P, "Excluding resource exporters", nores, z=z, n_rep=3),
            S(P, "High statistical capacity (SPI)", hi_spi, z=z, n_rep=3),
            S(P, "Outcome: GDP per capita growth", df, z=z, y="dgdppc", n_rep=3),
            S(P, "Population-weighted averages", df, z=z, weights="population", n_rep=3),
            S(P, "No Mundlak means", df, z=z, mundlak=False, n_rep=3),
            S(P, "WDI data, PWT 11.0 countries, 1996-2023", s_pwt11, z=z, n_rep=3),
        ]
    specs += [S("A", "Moderator: learning-adjusted schooling", df.dropna(subset=["lays_mean"]),
                z="lays_mean", n_rep=3),
              S("B", "Moderator: distance to US frontier", df, z="dist_us_l1", by_year=True,
                n_rep=3)]
    for P, zp in [("A", "hc_l1"), ("B", "lp_init")]:
        specs += [S(P, "PWT 11.0, 1996-2023", p11, z=zp, controls=PWT_BASE, n_rep=3),
                  S(P, "PWT 11.0, 1996-2019", p11[p11["year"] <= 2019], z=zp, controls=PWT_BASE,
                    n_rep=3),
                  S(P, "PWT 10.0, 1996-2019", p10, z=zp, controls=PWT_BASE, n_rep=3),
                  S(P, "PWT 11.0: outcome TFP growth", p11.dropna(subset=["dtfp"]), z=zp,
                    y="dtfp", controls=PWT_BASE, n_rep=3)]
    specs += [S("C", "Random forest nuisance", df, z="ysince_l1", learner=rf, n_rep=3),
              S("C", "Pre-COVID sample, 1996-2019", df[df["year"] <= 2019], z="ysince_l1", n_rep=3),
              S("C", "Adding mobile and broadband", df, z="ysince_l1",
                controls=BASE + ["mobile_l1", "broadband_l1"], n_rep=3)]
    # local projections: cumulative growth from t-1 to t+h
    for h in range(0, 9):
        specs.append(S("LP", f"h={h}", df.dropna(subset=[f"cum{h}"]), z="lp_init", y=f"cum{h}",
                       n_rep=3))
    # like-for-like broadband comparison with Czernich et al. (2011)
    hi_inc = df[df["income"] == "High income"]
    specs += [S("CZ", "High income, broadband, GDP per capita", hi_inc, z="hc_l1", d="broadband_l1",
                y="dgdppc", n_rep=3),
              S("CZ", "All countries, broadband, GDP per capita", df, z="hc_l1", d="broadband_l1",
                y="dgdppc", n_rep=3)]

    res = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(run_spec)(s) for s in specs)
    for s, (r, cv) in zip(specs, res):
        out.append(r)
        curves[f"{s['panel']}|{s['name']}"] = cv

    # linear benchmarks, dynamics, vintages, IV
    bb, Vb = fe_ols(df, df[["internet_l1"] + BASE].values, df["dlp"].values)
    extra["twfe"] = {"b": float(bb[0]), "se": float(np.sqrt(Vb[0, 0]))}
    extra["twfe_grad_lp"] = twfe_gradient(df, "lp_init")
    extra["twfe_grad_lp_spline"] = twfe_gradient(df, "lp_init", spline=True)
    extra["twfe_grad_hc"] = twfe_gradient(df, "hc_l1")
    grid = np.quantile(df["hc_l1"], np.linspace(0.01, 0.99, 33))
    bi, Vi = fe_ols(df, np.c_[df["internet_l1"], df["internet_l1"] * df["hc_l1"], df[BASE]],
                    df["dlp"].values)
    L = np.c_[np.ones(len(grid)), grid]
    extra["twfe_curve_hc"] = {"grid": grid.tolist(), "est": (L @ bi[:2]).tolist()}
    grid = np.quantile(df["lp_init"], np.linspace(0.01, 0.99, 33))
    bl, Vl = fe_ols(df, np.c_[df["internet_l1"], df["internet_l1"] * df["lp_init"], df[BASE]],
                    df["dlp"].values)
    L = np.c_[np.ones(len(grid)), grid]
    extra["twfe_curve_lp"] = {"grid": grid.tolist(), "est": (L @ bl[:2]).tolist()}
    extra["dose"] = dose_response(df)
    extra["five_year"] = five_year(df)
    extra["long_difference"] = long_difference(df)
    extra["iv"] = iv_check(df)
    extra["vintage"] = vintage_decomposition(p10, p11)
    cov = pd.read_csv(PROC / "coverage.csv")
    li = cov[cov["income"] == "Low income"]
    extra["low_income_included"] = sorted(li.loc[li["in_sample"], "country"].tolist())
    extra["low_income_excluded"] = sorted(li.loc[~li["in_sample"], "country"].tolist())
    extra["coverage_by_income"] = cov.groupby("income")["in_sample"].agg(["sum", "size"]).to_dict()
    scr = pd.read_csv(PROC / "screening_log.csv")
    extra["screening"] = {"n_flags": len(scr), "by_series": scr["series"].value_counts().to_dict()}
    extra["sample"] = {"n_obs": len(df), "n_countries": int(df["iso"].nunique()),
                       "years": [int(df["year"].min()), int(df["year"].max())]}

    pd.DataFrame(out).to_csv(RES / "empirical_summary.csv", index=False)
    json.dump({"curves": curves, "extra": extra}, open(RES / "empirical.json", "w"), indent=1,
              default=float)
    print(json.dumps(extra, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
