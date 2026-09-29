"""Monte Carlo study comparing Panel-DOSE with fixed-effects and machine-learning benchmarks.

Estimators
----------
TWFE, TWFE-interaction (quadratic D x Z), TWFE-spline (additive spline controls and a spline
in Z for the effect), FE-DML (constant effect), tuned boosted R-learner on the FE-DML residuals,
within-group DML sieve (nuisance learned on two-way demeaned data, in the spirit of Clarke and
Polselli), pooled DML sieve (no Mundlak features, no within step), two one-at-a-time ablations
(Mundlak features only; within step only), Panel-DOSE unpenalised and penalised, and an oracle
version of Panel-DOSE that uses the true nuisance functions (diagnostic only).
"""
import copy
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panel_dose import PanelDOSE, bspline_basis, within_two_way, _sup_t  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

CONTROLS = [f"x{j}" for j in range(1, 6)] + ["z"]
GRID = np.linspace(0.1, 0.9, 17)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / GH_W.sum()


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

    rho scales the strength of the nonlinear confounding in the outcome equation.  The columns
    starting with an underscore hold the true nuisance quantities used by the oracle estimator.
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
        idx0 = (-3.0 + 0.22 * t + 0.6 * alpha + 0.8 * np.sin(1.5 * x[:, 0])
                + 0.6 * x[:, 1] * (x[:, 2] > 0) + 1.5 * z)
        v = rng.standard_normal(N)
        d = 1 / (1 + np.exp(-(idx0 + v)))
        md = (1 / (1 + np.exp(-(idx0[:, None] + GH_X[None, :])))) @ GH_W
        g = rho * (1.5 * np.sin(1.5 * x[:, 0]) + 1.0 * x[:, 1] * (x[:, 2] > 0) + 0.5 * np.abs(x[:, 3])
                   + 0.8 * (z - 0.5) ** 2 * 4)
        th = theta_fun(shape, z)
        y = th * d + g + alpha + lam[t] + 0.5 * rng.standard_normal(N)
        rows.append(pd.DataFrame({"unit": np.arange(N), "t": t, "y": y, "d": d, "z": z,
                                  **{f"x{j + 1}": x[:, j] for j in range(5)},
                                  "_md": md, "_my": th * md + g + alpha + lam[t]}))
    return pd.concat(rows, ignore_index=True)


def lgbm():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15,
                             min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, n_jobs=1)


# ---------------------------------------------------------------------------- truth / metrics
def targets(shape, z):
    th = theta_fun(shape, z)
    q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
    zc = z - z.mean()
    return {"ate": th.mean(), "contrast": th[z > q2].mean() - th[z <= q1].mean(),
            "blp": np.mean(th * zc) / np.mean(zc ** 2)}


class LinearTheta:
    """theta(z) = L(z) gamma with a covariance matrix; used for the linear benchmarks."""

    def __init__(self, basis, coef, cov, z):
        self.basis, self.coef, self.cov, self.z = basis, coef, cov, z

    def _f(self, L):
        L = np.atleast_2d(L)
        return L @ self.coef, np.sqrt(np.clip(np.diag(L @ self.cov @ L.T), 0, None)), \
            L @ self.cov @ L.T

    def report(self):
        z = self.z
        q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
        B = self.basis(z)
        zc = z - z.mean()
        e, s, C = self._f(self.basis(GRID))
        crit, _, _ = _sup_t(e, C)
        out = {"est": e, "se": s, "ulo": e - crit * s, "uhi": e + crit * s}
        for k, l in [("ate", B.mean(0)), ("contrast", B[z > q2].mean(0) - B[z <= q1].mean(0)),
                     ("blp", (B * zc[:, None]).mean(0) / np.mean(zc ** 2))]:
            m, sd, _ = self._f(l)
            out[k], out[k + "_se"] = float(m[0]), float(sd[0])
        return out


def ols_fe(df, X, y):
    Xw = within_two_way(X, df["unit"], df["t"])
    yw = within_two_way(y, df["unit"], df["t"])
    b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    e = yw - Xw @ b
    u = pd.factorize(df["unit"])[0]
    S = np.zeros((u.max() + 1, Xw.shape[1]))
    np.add.at(S, u, Xw * e[:, None])
    A = np.linalg.pinv(Xw.T @ Xw)
    G, n, k = u.max() + 1, len(yw), Xw.shape[1]
    return b, G / (G - 1) * (n - 1) / max(n - k, 1) * A @ S.T @ S @ A


def twfe(df, kind):
    """Linear two-way FE: constant effect, quadratic interaction, or spline controls + spline effect."""
    z = df["z"].values
    d = df["d"].values
    if kind == "const":
        basis = lambda s: np.ones((len(s), 1))  # noqa: E731
        ctrl = df[CONTROLS].values
    elif kind == "quad":
        basis = lambda s: np.c_[np.ones(len(s)), s, s ** 2]  # noqa: E731
        ctrl = df[CONTROLS].values
    else:
        kn = np.quantile(z, [0.2, 0.4, 0.6, 0.8])
        basis = lambda s: bspline_basis(s, kn, z.min(), z.max())  # noqa: E731
        ctrl = np.hstack([bspline_basis(df[c].values, np.quantile(df[c], [0.25, .5, .75]),
                                        df[c].min(), df[c].max())[:, 1:] for c in CONTROLS])
    Bz = basis(z)
    X = np.c_[d[:, None] * Bz, ctrl]
    b, V = ols_fe(df, X, df["y"].values)
    k = Bz.shape[1]
    return LinearTheta(basis, b[:k], V[:k, :k], z).report()


def rlearner_tuned(z, ry, rd, unit):
    """Boosted R-learner (Nie and Wager, 2021) with hyper-parameters chosen by grouped CV."""
    keep = np.abs(rd) > 1e-3
    zt, tgt, w, u = z[keep, None], ry[keep] / rd[keep], rd[keep] ** 2, unit[keep]
    configs = [(3, 40, 100), (3, 40, 300), (7, 40, 200), (7, 80, 400), (15, 40, 200), (3, 80, 50)]
    err = np.zeros(len(configs))
    for tr, te in GroupKFold(5).split(zt, groups=u):
        for j, (nl, mc, ne) in enumerate(configs):
            m = lgb.LGBMRegressor(n_estimators=ne, learning_rate=0.05, num_leaves=nl,
                                  min_child_samples=mc, verbose=-1, n_jobs=1)
            m.fit(zt[tr], tgt[tr], sample_weight=w[tr])
            err[j] += np.sum(w[te] * (tgt[te] - m.predict(zt[te])) ** 2)
    nl, mc, ne = configs[int(np.argmin(err))]
    m = lgb.LGBMRegressor(n_estimators=ne, learning_rate=0.05, num_leaves=nl,
                          min_child_samples=mc, verbose=-1, n_jobs=1).fit(zt, tgt, sample_weight=w)
    th = m.predict(z[:, None])
    q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
    zc = z - z.mean()
    return {"est": m.predict(GRID[:, None]), "ate": th.mean(),
            "contrast": th[z > q2].mean() - th[z <= q1].mean(),
            "blp": np.mean(th * zc) / np.mean(zc ** 2)}


def dose_report(m):
    e = m.effect(GRID)
    z = m._Z
    q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
    out = {"est": e["est"].values, "se": e["se"].values, "ulo": e["ulo"].values,
           "uhi": e["uhi"].values}
    out["ate"], out["ate_se"] = m.average_effect()
    out["contrast"], out["contrast_se"] = m.group_contrast(z > q2, z <= q1)
    out["blp"], out["blp_se"] = m.blp_slope()
    out["lambda"], out["edf"] = m.lambda_, m.edf_
    if m.K_ > 1:
        out["p_const"] = m.shape_tests()["const"][1]
    return out


def from_raw(template, raws, within):
    """Panel-DOSE final stage on stored raw residuals, with or without the within step."""
    m = copy.copy(template)
    m._raw, m.within = raws, within
    m.rebasis(4, 3, penalty=True)
    return m


def wg_sieve(df):
    """Within-group DML sieve: learn the nuisance on two-way demeaned data (Clarke-Polselli)."""
    unit, t = df["unit"].values, df["t"].values
    Yw = within_two_way(df["y"].values, unit, t)
    Dw = within_two_way(df["d"].values, unit, t)
    Xw = within_two_way(df[CONTROLS].values, unit, t)
    ry, rd = np.empty(len(df)), np.empty(len(df))
    for tr, te in GroupKFold(5).split(Xw, groups=unit):
        ry[te] = Yw[te] - lgbm().fit(Xw[tr], Yw[tr]).predict(Xw[te])
        rd[te] = Dw[te] - lgbm().fit(Xw[tr], Dw[tr]).predict(Xw[te])
    m = PanelDOSE(lgbm, z_in_controls=True)
    m.unit_, m.time_ = unit, t
    m._Z, m._D, m._Y = df["z"].values, df["d"].values, df["y"].values
    m._Dw = Dw
    m.within = True
    m._raw = [(ry, rd)]
    m.rebasis(4, 3, penalty=True)
    return m


def oracle(df, template):
    ry = df["y"].values - df["_my"].values
    rd = df["d"].values - df["_md"].values
    return from_raw(template, [(ry, rd)], True)


def one_rep(shape, rep, N, T, rho, full=True):
    df = simulate(shape, N=N, T=T, seed=1000 * rep + 7, rho=rho)
    z = df["z"].values
    truth = theta_fun(shape, GRID)
    tg = targets(shape, z)
    out = []

    def rec(name, r):
        e = r["est"]
        row = {"shape": shape, "rep": rep, "N": N, "rho": rho, "method": name,
               "irmse": float(np.sqrt(np.mean((e - truth) ** 2))),
               "ibias": float(np.mean(e - truth))}
        for k in ["ate", "contrast", "blp"]:
            row[k + "_err"] = r[k] - tg[k]
            if k + "_se" in r:
                row[k + "_cover"] = float(abs(r[k] - tg[k]) <= 1.96 * r[k + "_se"])
        if "se" in r:
            row["cover"] = float(np.mean(np.abs(e - truth) <= 1.96 * r["se"]))
            row["ucover"] = float(np.all((truth >= r["ulo"]) & (truth <= r["uhi"])))
        for k in ["lambda", "edf", "p_const"]:
            if k in r:
                row[k] = r[k]
        out.append(row)

    rec("TWFE", twfe(df, "const"))
    rec("TWFE-interaction", twfe(df, "quad"))
    rec("TWFE-spline", twfe(df, "spline"))

    main = PanelDOSE(lgbm, n_rep=2, z_in_controls=True).fit(df, "y", "d", CONTROLS, "unit", "t",
                                                           z="z")
    const = copy.copy(main)
    const._raw = main._raw
    const.rebasis(0, 0, penalty=False)
    rec("FE-DML (constant)", dose_report(const))
    ryw = within_two_way(main._raw[0][0], main.unit_, main.time_)
    rdw = within_two_way(main._raw[0][1], main.unit_, main.time_)
    rec("FE-DML + R-learner (tuned)", rlearner_tuned(z, ryw, rdw, main.unit_))
    if full:
        rec("WG-DML sieve", dose_report(wg_sieve(df)))
        nomund = PanelDOSE(lgbm, n_rep=2, z_in_controls=True, mundlak=False).fit(
            df, "y", "d", CONTROLS, "unit", "t", z="z")
        rec("Pooled DML-sieve", dose_report(from_raw(nomund, nomund._raw, False)))
        rec("Ablation: Mundlak only", dose_report(from_raw(main, main._raw, False)))
        rec("Ablation: within only", dose_report(nomund))
    main.refit_final(penalty=False)
    rec("Panel-DOSE (unpenalised)", dose_report(main))
    main.refit_final(penalty=True)
    rec("Panel-DOSE", dose_report(main))
    if full:
        rec("Panel-DOSE (oracle nuisance)", dose_report(oracle(df, main)))
    return out


def main(reps=100, n_jobs=4):
    t0 = time.time()
    tasks = [(sh, r, 100, 24, 1.0, True) for sh in ["constant", "linear", "threshold", "hump"]
             for r in range(reps)]
    tasks += [("threshold", r, n, 24, 1.0, False) for n in (50, 200) for r in range(reps // 2)]
    tasks += [("threshold", r, 100, 24, rho, False) for rho in (0.25, 0.5, 2.0)
              for r in range(reps // 2)]
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(one_rep)(*a) for a in tasks)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(RES / "simulation_raw.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100,
         int(sys.argv[2]) if len(sys.argv) > 2 else 4)
