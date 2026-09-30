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
N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1


def lgbm():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                             min_child_samples=20, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, n_jobs=1)


def rf():
    return RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                 n_jobs=1, random_state=0)


# ----------------------------------------------------------------------------- estimation
N_MAIN, N_ROB = 20, 10   # repetitions of the sample split: main estimates / robustness checks


def fit_dose(df, y="dlp", d="internet_l1", z="hc_l1", controls=None, learner=lgbm, n_rep=N_MAIN,
             mode="vc", time="year", exclude=(), trend=None, fold_unit=None, **kw):
    controls = list(BASE if controls is None else controls)
    if mode == "vc" and z not in controls:
        controls.append(z)
    m = PanelDOSE(learner, mode=mode, n_rep=n_rep, z_in_controls=(mode == "vc"),
                  mundlak_exclude=STATIC + tuple(exclude), seed=kw.pop("seed", 11), **kw)
    m.fit(df.reset_index(drop=True), y, d, controls, "iso", time,
          z=z if mode == "vc" else None, trend=trend, fold_unit=fold_unit)
    return m


def tercile_masks(m, df, by_year=False):
    z = m._Z
    if by_year:
        r = df.groupby("year")[m._zname].rank(pct=True).values
        return r <= 1 / 3, (r > 1 / 3) & (r <= 2 / 3), r > 2 / 3
    q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
    zmin = z.min()
    if np.mean(z <= zmin) > 0.25:
        # mass point at the minimum (a moderator censored at zero): the mass point is the low
        # group and the remaining observations are split at their median
        q1 = zmin
        q2 = np.quantile(z[z > zmin], 0.5)
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
    need = [z, kw.get("y", "dlp"), kw.get("d", "internet_l1")] + list(kw.get("controls", BASE))
    kw.setdefault("n_rep", N_ROB)
    data = data.dropna(subset=need)
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
    m = PanelDOSE(lgbm, n_rep=N_ROB, n_knots=0, degree=0, penalty=False, z_in_controls=True,
                  n_folds=5, mundlak_exclude=STATIC, seed=11)
    m.fit(p, "dlp5", "internet_l1", controls + ["hc_l1"] if "hc_l1" not in controls else controls,
          "iso", "year", z="hc_l1")
    a = m.average_effect()
    return {"twfe": float(b[0]), "twfe_se": float(np.sqrt(V[0, 0])), "fedml": a[0],
            "fedml_se": a[1], "n_obs": len(p), "n_countries": int(p["iso"].nunique())}


def long_difference(df, start=2000, end=2024, controls=BASE, from_levels=False):
    """Cross-country long difference.

    Outcome: average annual productivity growth over start..end, i.e. 100 (ln LP_end -
    ln LP_{start-1}) / (end - start + 1), the mean of the annual growth rates of the panel.
    Regressor: the change in the panel treatment D_t = internet_{t-1} between start and end,
    i.e. internet_{end-1} - internet_{start-1}.  Controls: their panel values in the start year
    (lagged, i.e. mostly 1999 values).  OLS (HC1) and cross-fitted partially linear DML."""
    if from_levels:
        # every country with productivity in 1999 and 2024 and the start-year controls,
        # without the panel's trimming and 15-year rule
        df = pd.read_csv(PROC / "panel_all.csv")
    a = df[df["year"] == start].set_index("iso")
    b = df[df["year"] == end].set_index("iso")
    raw = pd.read_csv(ROOT / "data" / "raw" / "wdi" / "gdp_per_worker.csv")
    lp = np.log(raw.pivot(index="iso", columns="year", values="gdp_per_worker"))
    ids = a.index.intersection(b.index).intersection(lp.index)
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


# ----------------------------------------------------------------------------- exogeneity
def fedml(df, **kw):
    """Constant-effect FE-DML estimate (and s.e.) from the Panel-DOSE residuals."""
    m = fit_dose(df, **kw)
    c = copy.copy(m)
    c._raw = m._raw
    c.rebasis(0, 0, penalty=False)
    return c.average_effect()


def lead_test(df, controls=BASE):
    """Timing checks.

    * Strict-exogeneity check (Wooldridge, 2010, Section 10.7): add the lead D_{t+1} =
      internet_t to the static model; under strict exogeneity its coefficient is zero.
    * Lead as the only treatment (FE-DML) and TWFE with the adoption share dated t-6 ... t+4,
      each on the common subsample where all dates are observed.
    * The lagged change D_t - D_{t-1} as treatment.
    """
    d = df.dropna(subset=["internet_f1"]).reset_index(drop=True)
    b, V = fe_ols(d, d[["internet_l1", "internet_f1"] + controls].values, d["dlp"].values)
    lead, lead_se = fedml(d, d="internet_f1", z="hc_l1", controls=controls + ["internet_l1"],
                          n_rep=N_ROB)
    lead_only, lead_only_se = fedml(d, d="internet_f1", z="hc_l1", n_rep=N_ROB)
    out = {"twfe_d": float(b[0]), "twfe_d_se": float(np.sqrt(V[0, 0])),
           "twfe_lead": float(b[1]), "twfe_lead_se": float(np.sqrt(V[1, 1])),
           "fedml_lead": lead, "fedml_lead_se": lead_se,
           "fedml_lead_only": lead_only, "fedml_lead_only_se": lead_only_se,
           "n_obs": len(d), "n_countries": int(d["iso"].nunique())}
    # adoption dated s = t-6, ..., t+4 (D_t = internet_{t-1}); common subsample
    raw = pd.read_csv(PROC / "panel_all.csv")[["iso", "year", "internet_l1"]]
    x = df.copy()
    shifts = list(range(-5, 6))  # internet_{t-1+k}: k = -5 ... 5
    for k in shifts:
        r = raw.copy()
        r["year"] = r["year"] - k
        x = x.merge(r.rename(columns={"internet_l1": f"net_{k}"}), on=["iso", "year"], how="left")
    x = x.dropna(subset=[f"net_{k}" for k in shifts]).reset_index(drop=True)
    timing = {}
    for k in shifts:
        bk, Vk = fe_ols(x, x[[f"net_{k}"] + controls].values, x["dlp"].values)
        timing[str(k)] = [float(bk[0]), float(np.sqrt(Vk[0, 0]))]
    out["timing"] = timing
    out["timing_n"] = len(x)
    # lagged change
    df2 = df.dropna(subset=["internet_l2"]).copy()
    df2["dnet"] = df2["internet_l1"] - df2["internet_l2"]
    b2, V2 = fe_ols(df2, df2[["dnet"] + controls].values, df2["dlp"].values)
    ch, ch_se = fedml(df2, d="dnet", z="hc_l1", n_rep=N_ROB)
    out.update({"twfe_change": float(b2[0]), "twfe_change_se": float(np.sqrt(V2[0, 0])),
                "fedml_change": ch, "fedml_change_se": ch_se})
    return out


def twfe_checks(df, controls=BASE):
    """Linear two-way FE checks against trends, convergence and development-specific shocks."""
    out = {}
    y = df["dlp"].values
    X = df[["internet_l1"] + controls].values

    def fit(Xm, yv, d=df, trend=None):
        Xw = within_two_way(Xm, d["iso"], d["year"], trend=trend)
        yw = within_two_way(yv, d["iso"], d["year"], trend=trend)
        b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        e = yw - Xw @ b
        u = pd.factorize(d["iso"])[0]
        S_ = np.zeros((u.max() + 1, Xw.shape[1]))
        np.add.at(S_, u, Xw * e[:, None])
        A = np.linalg.pinv(Xw.T @ Xw)
        G, n, k = u.max() + 1, len(yw), Xw.shape[1]
        Vv = G / (G - 1) * (n - 1) / max(n - k, 1) * A @ S_.T @ S_ @ A
        return float(b[0]), float(np.sqrt(Vv[0, 0]))

    yr = df["year"].values.astype(float)
    out["baseline"] = fit(X, y)
    out["linear_trends"] = fit(X, y, trend=yr)
    # quadratic trends: partial out unit x year^2 as extra regressors after linear trends
    q = pd.get_dummies(df["iso"]).values.astype(float) * ((yr - 2010) ** 2)[:, None]
    out["quadratic_trends"] = fit(np.c_[X, q], y, trend=yr)
    out["lagged_productivity"] = fit(np.c_[X, df["lp_l1"].values], y)
    tl = pd.get_dummies(pd.qcut(df["lp_init"], 3, labels=False).astype(str) + "_"
                        + df["year"].astype(str)).values.astype(float)
    out["lp_init_tercile_x_year"] = fit(np.c_[X, tl], y)
    ry = pd.get_dummies(df["ryear"]).values.astype(float)
    out["region_x_year"] = fit(np.c_[X, ry], y)
    for P, per in [("1996-2011", df["year"] <= 2011), ("2012-2025", df["year"] >= 2012)]:
        d = df[per].reset_index(drop=True)
        out[f"period_{P}"] = fit(d[["internet_l1"] + controls].values, d["dlp"].values, d=d)
    d = df[~df["transition"]].reset_index(drop=True)
    out["no_transition"] = fit(d[["internet_l1"] + controls].values, d["dlp"].values, d=d)
    # decomposition dlp = dgdp - demp (WDI GDP growth and implied employment growth)
    d = df.dropna(subset=["dgdp", "demp"]).reset_index(drop=True)
    Xd = d[["internet_l1"] + controls].values
    out["outcome_gdp_growth"] = fit(Xd, d["dgdp"].values, d=d)
    out["outcome_employment_growth"] = fit(Xd, d["demp"].values, d=d)
    out["outcome_emp_rate_growth"] = fit(Xd, (d["dgdppc"] - d["dlp"]).values, d=d)
    return out


def twfe_pwt(p, controls=PWT_BASE):
    """Linear benchmarks along initial productivity in a PWT panel."""
    return {"interaction": twfe_gradient(p, "lp_init", controls=controls),
            "spline": twfe_gradient(p, "lp_init", controls=controls, spline=True)}


def dose_response(df):
    """Dose-response model f(D) in internet penetration (O'Sullivan penalty; R13 re-examination)."""
    m = PanelDOSE(lgbm, mode="dose", n_rep=N_MAIN, mundlak_exclude=STATIC, seed=11)
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
    common = pd.read_csv(PROC / "panel_common.csv")
    added = sorted(set(df["iso"]) - set(common["iso"]))
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
    w = np.linalg.eigh(np.cov(zs.dropna().T))[1][:, -1]
    pc = pd.Series(zs.values @ (w * np.sign(w.sum())), index=df.index)
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
        specs.append(S(P, "Baseline", df, z=z, n_rep=N_MAIN))
        specs.append(S(P, "Complete-ICT sample", common, z=z, n_rep=N_MAIN))
        specs.append(S(P, "Country-specific linear trends", df, z=z, trend="year"))
        specs.append(S(P, "Excluding transition economies", df[~df["transition"]], z=z))
        specs.append(S(P, "Excluding the eight countries outside the complete-ICT sample",
                       df[~df["iso"].isin(added)], z=z))
        specs.append(S(P, "Period 1996-2011", df[df["year"] <= 2011], z=z))
        specs.append(S(P, "Period 2012-2025", df[df["year"] >= 2012], z=z))
        specs.append(S(P, "Region-by-year effects", df, z=z, time="ryear",
                       controls=BASE + ["year", "region_code"], exclude=("year", "region_code"),
                       n_rep=N_ROB))
    for P, z in [("A", "hc_l1"), ("B", "lp_init")]:
        specs += [
            S(P, "Original specification", df, z=z, controls=ORIGINAL + (["lp_init"] if z == "lp_init" else [])),
            S(P, "Adding the investment share", df, z=z, controls=BASE + ["csh_i_l1"]),
            S(P, "Adding mobile and broadband", df, z=z, controls=BASE + ["mobile_l1", "broadband_l1"]),
            S(P, "Adding mobile and broadband, no zero-filling", df, z=z,
              controls=BASE + ["mobile_l1", "broadband_nf_l1"], n_rep=N_ROB),
            S(P, "Treatment: composite ICT index", df, z=z, d="ict_index"),
            S(P, "Random forest nuisance", df, z=z, learner=rf, n_rep=N_ROB),
            S(P, "Unpenalised sieve", df, z=z, penalty=False, n_rep=N_ROB),
            S(P, "Three interior knots", df, z=z, n_knots=3, n_rep=N_ROB),
            S(P, "GCV penalty", df, z=z, select="gcv", n_rep=N_ROB),
            S(P, "Two-year lag of internet", df, z=z, d="internet_l2", n_rep=N_ROB),
            S(P, "Pre-COVID sample, 1996-2019", df[df["year"] <= 2019], z=z, n_rep=N_ROB),
            S(P, "Excluding 2020-2021", df[~df["year"].isin([2020, 2021])], z=z, n_rep=N_ROB),
            S(P, "Excluding 2025", df[df["year"] <= 2024], z=z, n_rep=N_ROB),
            S(P, "Untrimmed outcome", unt, z=z, n_rep=N_ROB),
            S(P, "Winsorised outcome", wins, z=z, n_rep=N_ROB),
            S(P, "Without data screening", df_scr_off, z=z, n_rep=N_ROB),
            S(P, "Excluding resource exporters", nores, z=z, n_rep=N_ROB),
            S(P, "High statistical capacity (SPI)", hi_spi, z=z, n_rep=N_ROB),
            S(P, "Outcome: GDP per capita growth", df, z=z, y="dgdppc", n_rep=N_ROB),
            S(P, "Population-weighted averages", df, z=z, weights="population", n_rep=N_ROB),
            S(P, "No Mundlak means", df, z=z, mundlak=False, n_rep=N_ROB),
            S(P, "WDI data, PWT 11.0 countries, 1996-2023", s_pwt11, z=z, n_rep=N_MAIN),
        ]
    specs += [S("A", "Moderator: learning-adjusted schooling", df.dropna(subset=["lays_mean"]),
                z="lays_mean", n_rep=N_ROB),
              S("B", "Moderator: distance to US frontier", df, z="dist_us_l1", by_year=True,
                n_rep=N_ROB),
              S("B", "Countries with pre-sample productivity only", df[df["lp_init_pre"]],
                z="lp_init", n_rep=N_ROB)]
    for P, zp in [("A", "hc_l1"), ("B", "lp_init")]:
        specs += [S(P, "PWT 11.0, 1996-2023", p11, z=zp, controls=PWT_BASE, n_rep=N_MAIN),
                  S(P, "PWT 11.0, 1996-2019", p11[p11["year"] <= 2019], z=zp, controls=PWT_BASE,
                    n_rep=N_ROB),
                  S(P, "PWT 10.0, 1996-2019", p10, z=zp, controls=PWT_BASE, n_rep=N_MAIN),
                  S(P, "PWT 11.0: outcome TFP growth", p11.dropna(subset=["dtfp"]), z=zp,
                    y="dtfp", controls=PWT_BASE, n_rep=N_ROB)]
    na_ctrl = [c if c != "lp_init" else "lp_init_na" for c in PWT_BASE]
    specs += [S("B", "PWT 11.0, national-prices moderator", p11, z="lp_init_na",
                controls=na_ctrl, n_rep=N_ROB),
              S("B", "PWT 10.0, national-prices moderator", p10, z="lp_init_na",
                controls=na_ctrl, n_rep=N_ROB)]
    specs += [S("C", "Random forest nuisance", df, z="ysince_l1", learner=rf, n_rep=N_ROB),
              S("C", "Pre-COVID sample, 1996-2019", df[df["year"] <= 2019], z="ysince_l1", n_rep=N_ROB),
              S("C", "Adding mobile and broadband", df, z="ysince_l1",
                controls=BASE + ["mobile_l1", "broadband_l1"], n_rep=N_ROB),
              S("C", "Predetermined moderator (zero before take-off)", df, z="ysince_pre_l1",
                n_rep=N_ROB)]
    # local projections: cumulative growth from t-1 to t+h
    for h in range(0, 9):
        specs.append(S("LP", f"h={h}", df.dropna(subset=[f"cum{h}"]), z="lp_init", y=f"cum{h}",
                       n_rep=N_ROB))
        # common sample: observations for which the longest horizon is observed
        specs.append(S("LPc", f"h={h}", df.dropna(subset=["cum8"]), z="lp_init", y=f"cum{h}",
                       n_rep=N_ROB))
    # like-for-like broadband comparison with Czernich et al. (2011)
    hi_inc = df[df["income"] == "High income"]
    specs += [S("CZ", "High income, broadband, GDP per capita", hi_inc, z="hc_l1", d="broadband_l1",
                y="dgdppc", n_rep=N_ROB),
              S("CZ", "All countries, broadband, GDP per capita", df, z="hc_l1", d="broadband_l1",
                y="dgdppc", n_rep=N_ROB)]

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
    extra["long_difference_levels"] = long_difference(df, from_levels=True)
    extra["twfe_checks"] = twfe_checks(df)
    extra["twfe_pwt11"] = twfe_pwt(p11)
    extra["twfe_pwt10"] = twfe_pwt(p10)
    extra["added_countries"] = added
    extra["iv"] = iv_check(df)
    extra["lead_test"] = lead_test(df)
    cov = pd.read_csv(PROC / "coverage.csv")
    li = cov[cov["income"] == "Low income"]
    extra["low_income_included"] = sorted(li.loc[li["in_sample"], "country"].tolist())
    extra["low_income_excluded"] = sorted(li.loc[~li["in_sample"], "country"].tolist())
    extra["coverage_by_income"] = cov.groupby("income")["in_sample"].agg(["sum", "size"]).to_dict()
    scr = pd.read_csv(PROC / "screening_log.csv")
    extra["screening"] = {"n_flags": len(scr), "by_series": scr["series"].value_counts().to_dict()}
    extra["sample"] = {"n_obs": len(df), "n_countries": int(df["iso"].nunique()),
                       "years": [int(df["year"].min()), int(df["year"].max())],
                       "n_pre_sample": int(df.loc[df["lp_init_pre"], "iso"].nunique()),
                       "no_pre_sample": sorted(df.loc[~df["lp_init_pre"], "country"].unique()),
                       "common_obs": len(common), "common_countries": int(common["iso"].nunique()),
                       "untrimmed_obs": len(unt), "untrimmed_countries": int(unt["iso"].nunique()),
                       "unscreened_obs": len(df_scr_off),
                       "pwt10": [len(p10), int(p10["iso"].nunique())],
                       "pwt11": [len(p11), int(p11["iso"].nunique())]}

    pd.DataFrame(out).to_csv(RES / "empirical_summary.csv", index=False)
    json.dump({"curves": curves, "extra": extra}, open(RES / "empirical.json", "w"), indent=1,
              default=float)
    print(json.dumps(extra, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
