"""Monte Carlo study comparing Panel-DOSE with fixed-effects and machine-learning benchmarks."""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_dose import PanelDOSE, within_two_way  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

CONTROLS = [f"x{j}" for j in range(1, 6)] + ["z"]
GRID = np.linspace(0.1, 0.9, 17)


def theta_fun(shape, z):
    if shape == "constant":
        return np.full_like(z, 1.0)
    if shape == "linear":
        return -0.5 + 3.0 * z
    if shape == "threshold":
        return -0.5 + 2.5 / (1 + np.exp(-12 * (z - 0.5)))
    if shape == "hump":
        return 2.0 - 8.0 * (z - 0.5) ** 2
    raise ValueError(shape)


def simulate(shape, N=100, T=24, seed=0, rho=1.0):
    """Country-year panel with fixed-effect and nonlinear confounding of digital adoption.

    rho scales the strength of the nonlinear confounding in the outcome equation.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.standard_normal(N)
    lam = np.cumsum(0.2 * rng.standard_normal(T))
    rows = []
    x = rng.standard_normal((N, 5)) + 0.5 * alpha[:, None]
    zi = rng.uniform(0, 1, N) * 0.6 + 0.15 * (alpha > 0)
    for t in range(T):
        x = 0.6 * x + 0.4 * (rng.standard_normal((N, 5)) + 0.5 * alpha[:, None])
        z = np.clip(zi + 0.012 * t + 0.05 * rng.standard_normal(N), 0, 1)
        idx = (-3.0 + 0.22 * t + 0.6 * alpha + 0.8 * np.sin(1.5 * x[:, 0]) + 0.6 * x[:, 1] * (x[:, 2] > 0)
               + 1.5 * z + 1.0 * rng.standard_normal(N))
        d = 1 / (1 + np.exp(-idx))
        g = rho * (1.5 * np.sin(1.5 * x[:, 0]) + 1.0 * x[:, 1] * (x[:, 2] > 0) + 0.5 * np.abs(x[:, 3])
                   + 0.8 * (z - 0.5) ** 2 * 4)
        y = theta_fun(shape, z) * d + g + alpha + lam[t] + 0.5 * rng.standard_normal(N)
        rows.append(pd.DataFrame({"unit": np.arange(N), "t": t, "y": y, "d": d, "z": z,
                                  **{f"x{j + 1}": x[:, j] for j in range(5)}}))
    return pd.concat(rows, ignore_index=True)


def lgbm():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15,
                             min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, n_jobs=1)


# ---------------------------------------------------------------------------- benchmarks
def twfe(df, interact):
    """Linear two-way fixed effects with (optionally) a quadratic interaction D x Z."""
    cols = ["d"] + CONTROLS
    X = df[cols].copy()
    if interact:
        X["dz"] = df["d"] * df["z"]
        X["dz2"] = df["d"] * df["z"] ** 2
    Xw = within_two_way(X.values, df["unit"], df["t"])
    yw = within_two_way(df["y"].values, df["unit"], df["t"])
    b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    e = yw - Xw @ b
    u = pd.factorize(df["unit"])[0]
    S = np.zeros((u.max() + 1, Xw.shape[1]))
    np.add.at(S, u, Xw * e[:, None])
    A = np.linalg.pinv(Xw.T @ Xw)
    V = A @ S.T @ S @ A
    if interact:
        L = np.zeros((len(GRID), Xw.shape[1]))
        L[:, 0], L[:, -2], L[:, -1] = 1, GRID, GRID ** 2
    else:
        L = np.zeros((len(GRID), Xw.shape[1]))
        L[:, 0] = 1
    est = L @ b
    se = np.sqrt(np.diag(L @ V @ L.T))
    zbar = df["z"].values
    lavg = np.array([1, *([0] * len(CONTROLS))] + ([zbar.mean(), (zbar ** 2).mean()] if interact else []))
    return est, se, float(lavg @ b)


def rlearner(df):
    """FE-DML residuals followed by a boosted R-learner for theta(z) (no inference)."""
    est = PanelDOSE(lgbm, mode="vc", n_knots=0, degree=0, n_rep=1, penalty=False,
                    z_in_controls=True)
    # obtain residuals from the same orthogonalisation as Panel-DOSE with a single basis term
    df = df.reset_index(drop=True)
    feats = pd.concat([df[CONTROLS], df.groupby("unit")[CONTROLS + ["d"]].transform("mean")
                       .add_suffix("_bar")], axis=1)
    feats["_t"] = df["t"]
    F = feats.values
    est.unit_ = df["unit"].values
    est._setup_basis(df["z"].values)
    Phi = df["d"].values[:, None]
    ry, rd = est._crossfit(F, df["y"].values, df["d"].values, df["z"].values, Phi,
                           df["unit"].values)
    ry = within_two_way(ry, df["unit"], df["t"])
    rd = within_two_way(rd[:, 0], df["unit"], df["t"])
    keep = np.abs(rd) > 1e-3
    m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=40,
                          verbose=-1, n_jobs=1)
    m.fit(df.loc[keep, ["z"]], ry[keep] / rd[keep], sample_weight=rd[keep] ** 2)
    est_g = m.predict(pd.DataFrame({"z": GRID}))
    avg = float(m.predict(df[["z"]]).mean())
    return est_g, avg


def dose_variant(df, **kw):
    m = PanelDOSE(lgbm, mode="vc", n_rep=kw.pop("n_rep", 2), z_in_controls=True, **kw)
    m.fit(df, "y", "d", CONTROLS, "unit", "t", z="z")
    eff = m.effect(GRID)
    avg, _ = m.average_effect()
    return eff["est"].values, eff["se"].values, avg, eff


def one_rep(shape, rep, N, T, rho):
    df = simulate(shape, N=N, T=T, seed=1000 * rep + 7, rho=rho)
    truth = theta_fun(shape, GRID)
    ate = theta_fun(shape, df["z"].values).mean()
    out = []

    def rec(name, est, avg, se=None, uni=None):
        r = {"shape": shape, "rep": rep, "N": N, "rho": rho, "method": name,
             "irmse": float(np.sqrt(np.mean((est - truth) ** 2))),
             "ibias": float(np.mean(est - truth)), "ate_err": avg - ate}
        if se is not None:
            r["cover"] = float(np.mean(np.abs(est - truth) <= 1.96 * se))
        if uni is not None:
            r["ucover"] = float(np.all((truth >= uni[0]) & (truth <= uni[1])))
        out.append(r)

    e, s, a = twfe(df, False)
    rec("TWFE", e, a, s)
    e, s, a = twfe(df, True)
    rec("TWFE-interaction", e, a, s)
    # constant-effect FE-DML (Clarke & Polselli-type within-group DML)
    m = PanelDOSE(lgbm, mode="vc", n_knots=0, degree=0, n_rep=2, penalty=False, z_in_controls=True)
    m.fit(df, "y", "d", CONTROLS, "unit", "t", z="z")
    ef = m.effect(GRID)
    rec("FE-DML (constant)", ef["est"].values, m.average_effect()[0], ef["se"].values)
    e, a = rlearner(df)
    rec("FE-DML + R-learner", e, a)
    e, s, a, _ = dose_variant(df, mundlak=False, within=False)
    rec("Pooled DML-sieve", e, a, s)
    m = PanelDOSE(lgbm, mode="vc", n_rep=2, z_in_controls=True)
    m.fit(df, "y", "d", CONTROLS, "unit", "t", z="z")
    for name, pen in [("Panel-DOSE", True), ("Panel-DOSE (unpenalised)", False)]:
        m.refit_final(penalty=pen)
        ef = m.effect(GRID)
        rec(name, ef["est"].values, m.average_effect()[0], ef["se"].values,
            (ef["ulo"].values, ef["uhi"].values))
    return out


def main(reps=100, n_jobs=3):
    t0 = time.time()
    tasks = [(sh, r, 100, 24, 1.0) for sh in ["constant", "linear", "threshold", "hump"]
             for r in range(reps)]
    # sensitivity: sample size and confounding strength, threshold design
    tasks += [("threshold", r, n, 24, 1.0) for n in (50, 200) for r in range(reps // 2)]
    tasks += [("threshold", r, 100, 24, rho) for rho in (0.5, 2.0) for r in range(reps // 2)]
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one_rep)(*a) for a in tasks)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(RES / "simulation_raw.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
