"""Empirical application: heterogeneous productivity returns to internet adoption, 1996-2019."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_dose import PanelDOSE, within_two_way  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

BASE_CONTROLS = ["lp_l1", "kl_l1", "hc_l1", "csh_i_l1", "csh_g_l1", "open_l1", "dpop_l1",
                 "labsh_l1", "dlp_l1", "mobile_l1", "broadband_l1"]


def lgbm():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                             min_child_samples=20, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, n_jobs=1)


def rf():
    return RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                 n_jobs=1, random_state=0)


def twfe(df, y, d, controls, inter=None):
    cols = [d] + controls
    X = df[cols].copy()
    if inter is not None:
        X["dz"] = df[d] * df[inter]
    Xw = within_two_way(X.values, df["iso"], df["year"])
    yw = within_two_way(df[y].values, df["iso"], df["year"])
    b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    e = yw - Xw @ b
    u = pd.factorize(df["iso"])[0]
    S = np.zeros((u.max() + 1, Xw.shape[1]))
    np.add.at(S, u, Xw * e[:, None])
    A = np.linalg.pinv(Xw.T @ Xw)
    G, n, k = u.max() + 1, len(yw), Xw.shape[1]
    V = G / (G - 1) * (n - 1) / (n - k) * A @ S.T @ S @ A
    return b, np.sqrt(np.diag(V)), V


def fit_dose(df, y="dlp", d="internet_l1", z="hc_l1", controls=None, learner=lgbm, mode="vc",
             **kw):
    controls = list(BASE_CONTROLS if controls is None else controls)
    if mode == "vc" and z not in controls:
        controls.append(z)
    m = PanelDOSE(learner, mode=mode, n_rep=kw.pop("n_rep", 5), z_in_controls=(mode == "vc"),
                  seed=kw.pop("seed", 11), **kw)
    m.fit(df, y, d, controls, "iso", "year", z=z if mode == "vc" else None)
    return m


def summarise(m, name, grid):
    avg, se = m.average_effect()
    w1 = m.wald(1)
    w2 = m.wald(2) if m.mode == "vc" else (np.nan, 0, np.nan)
    eff = m.effect(grid)
    r = {"spec": name, "avg": avg, "avg_se": se, "wald_const": w1[0], "df_const": w1[1],
         "p_const": w1[2], "wald_lin": w2[0], "df_lin": w2[1], "p_lin": w2[2],
         "lambda": m.lambda_, "eff_lo_end": eff["est"].iloc[0], "eff_hi_end": eff["est"].iloc[-1]}
    if m.mode == "vc":
        z = m._Z
        q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
        for lab, mask in [("low", z <= q1), ("mid", (z > q1) & (z <= q2)), ("high", z > q2)]:
            r[f"gate_{lab}"], r[f"gate_{lab}_se"] = m.group_effect(mask)
        r["gate_diff"], r["gate_diff_se"] = m.group_contrast(z > q2, z <= q1)
        r["gate_diff_p"] = float(2 * stats.norm.sf(abs(r["gate_diff"] / r["gate_diff_se"])))
        # best linear projection of theta(Z) on (1, Z): 1-df test of heterogeneity
        kn, dg, pen = m.n_knots, m.degree, m.penalty
        m.rebasis(0, 1, penalty=False)
        c = np.array([-1.0, 1.0]) / (m.hi_ - m.lo_)
        r["blp_slope"] = float(c @ m.coef_)
        r["blp_slope_se"] = float(np.sqrt(c @ m.cov_ @ c))
        r["blp_p"] = float(2 * stats.norm.sf(abs(r["blp_slope"] / r["blp_slope_se"])))
        m.rebasis(kn, dg, penalty=pen)
    return r


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    out, curves = [], {}

    # ---------------- descriptive statistics
    desc_cols = ["dlp", "dtfp", "internet_l1", "mobile_l1", "broadband_l1", "hc_l1", "lp_l1",
                 "kl_l1", "csh_i_l1", "csh_g_l1", "open_l1", "dpop_l1", "labsh_l1"]
    desc = df[desc_cols].describe().T[["count", "mean", "std", "min", "max"]]
    desc.to_csv(RES / "descriptives.csv")

    # ---------------- linear benchmarks
    b, s, _ = twfe(df, "dlp", "internet_l1", BASE_CONTROLS)
    bi, si, Vi = twfe(df, "dlp", "internet_l1", BASE_CONTROLS, inter="hc_l1")
    lin = {"twfe_b": b[0], "twfe_se": s[0], "twfe_int_b": bi[0], "twfe_int_se": si[0],
           "twfe_int_dz": bi[-1], "twfe_int_dz_se": si[-1]}
    hc_grid = np.quantile(df["hc_l1"], np.linspace(0.05, 0.95, 19))
    Lg = np.zeros((len(hc_grid), len(bi)))
    Lg[:, 0], Lg[:, -1] = 1, hc_grid
    curves["twfe_hc"] = {"grid": hc_grid.tolist(), "est": (Lg @ bi).tolist(),
                         "se": np.sqrt(np.diag(Lg @ Vi @ Lg.T)).tolist()}
    constant = fit_dose(df, n_knots=0, degree=0, penalty=False)
    lin["fedml_b"], lin["fedml_se"] = constant.average_effect()

    # ---------------- main specification: theta(human capital)
    main_m = fit_dose(df)
    out.append(summarise(main_m, "Baseline (LightGBM, Z = human capital)", hc_grid))
    e = main_m.effect(hc_grid)
    curves["dose_hc"] = e.to_dict(orient="list")
    # where does the return become significantly positive?
    sig = e[e["lo"] > 0]
    lin["hc_threshold_pointwise"] = float(sig["grid"].min()) if len(sig) else None
    lin["hc_share_obs_above_threshold"] = (float((df["hc_l1"] >= sig["grid"].min()).mean())
                                           if len(sig) else None)

    # ---------------- theta(digital maturity): dose-response in internet penetration
    dose_m = fit_dose(df, mode="dose", z=None)
    net_grid = np.quantile(df["internet_l1"], np.linspace(0.05, 0.95, 19))
    out.append(summarise(dose_m, "Dose response in internet penetration", net_grid))
    curves["dose_net"] = dose_m.effect(net_grid).to_dict(orient="list")

    # ---------------- theta(development level)
    lp_grid = np.quantile(df["lp_l1"], np.linspace(0.05, 0.95, 19))
    dev_m = fit_dose(df, z="lp_l1")
    out.append(summarise(dev_m, "Z = initial labour productivity", lp_grid))
    curves["dose_lp"] = dev_m.effect(lp_grid).to_dict(orient="list")

    # ---------------- robustness on the baseline
    rob = [
        ("Random forest nuisance", dict(learner=rf)),
        ("Three interior knots", dict(n_knots=3)),
        ("Six interior knots", dict(n_knots=6)),
        ("Unpenalised sieve", dict(penalty=False)),
        ("Two-year lag of internet", dict(d="internet_l2")),
        ("Without lagged growth", dict(controls=[c for c in BASE_CONTROLS if c != "dlp_l1"])),
        ("Broadband as treatment", dict(d="broadband_l1",
                                        controls=[c for c in BASE_CONTROLS if c != "broadband_l1"]
                                        + ["internet_l1"])),
        ("Outcome: TFP growth", dict(y="dtfp", controls=[c for c in BASE_CONTROLS if c != "dlp_l1"])),
        ("No Mundlak means", dict(mundlak=False)),
    ]
    for name, kw in rob:
        sub = df.dropna(subset=["dtfp"]) if kw.get("y") == "dtfp" else df
        m = fit_dose(sub, **kw)
        out.append(summarise(m, name, hc_grid))
        curves["rob_" + name] = m.effect(hc_grid).to_dict(orient="list")
        print(name, "done")

    rob_lp = [
        ("Z = productivity: random forest nuisance", dict(learner=rf)),
        ("Z = productivity: two-year lag of internet", dict(d="internet_l2")),
        ("Z = productivity: without lagged growth",
         dict(controls=[c for c in BASE_CONTROLS if c != "dlp_l1"])),
        ("Z = productivity: outcome TFP growth",
         dict(y="dtfp", controls=[c for c in BASE_CONTROLS if c != "dlp_l1"])),
    ]
    for name, kw in rob_lp:
        sub = df.dropna(subset=["dtfp"]) if kw.get("y") == "dtfp" else df
        m = fit_dose(sub, z="lp_l1", **kw)
        out.append(summarise(m, name, lp_grid))
        curves["roblp_" + name] = m.effect(lp_grid).to_dict(orient="list")
        print(name, "done")

    pd.DataFrame(out).to_csv(RES / "empirical_summary.csv", index=False)
    json.dump({"linear": lin, "curves": curves, "n_obs": len(df),
               "n_countries": int(df["iso"].nunique()),
               "years": [int(df["year"].min()), int(df["year"].max())]},
              open(RES / "empirical.json", "w"), indent=1, default=float)
    print(pd.DataFrame(out).round(3).to_string())
    print(json.dumps(lin, indent=1, default=float))


if __name__ == "__main__":
    main()
