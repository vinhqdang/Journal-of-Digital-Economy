"""Panel-DOSE: a Debiased Orthogonal Sieve Estimator for heterogeneous effects in panels.

Target model
------------
    Y_it = theta(Z_it) * D_it + g(X_it) + alpha_i + lambda_t + e_it        (varying-coefficient mode)
    Y_it = f(D_it)            + g(X_it) + alpha_i + lambda_t + e_it        (dose-response mode)

theta(.) or f(.) is approximated by a penalised cubic B-spline sieve, phi(D, Z)'beta, while
g(.) is left unrestricted and learned with machine learning.  The algorithm

1. augments the controls with Mundlak (within-unit) means so that the nuisance learners can
   absorb the part of the fixed effect that is correlated with observables;
2. cross-fits the nuisance functions E[Y | F] and E[D | F] (or E[phi_k | F]) with folds formed
   by *units*, so no country is used to predict itself;
3. two-way demeans the cross-fitted residuals;
4. solves the Neyman-orthogonal moment for beta with an O'Sullivan penalty (the integrated
   squared second derivative of theta), whose null space is exactly the linear functions for
   any knot placement; the penalty weight is chosen by leave-units-out cross-validation;
5. repeats the sample split and aggregates each reported functional (or vector of functionals)
   by the median across splits, with a cluster-robust sandwich variance, and uses the same
   functional-wise aggregation for sup-t simultaneous bands and shape tests.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import BSpline
from sklearn.model_selection import GroupKFold


# ----------------------------------------------------------------------------- basis
def bspline_basis(x, knots_interior, lo, hi, degree=3, deriv=0):
    t = np.r_[[lo] * (degree + 1), knots_interior, [hi] * (degree + 1)]
    k = len(t) - degree - 1
    x = np.clip(np.asarray(x, float), lo, hi)
    out = np.empty((len(x), k))
    for j in range(k):
        c = np.zeros(k)
        c[j] = 1.0
        b = BSpline(t, c, degree, extrapolate=False)
        out[:, j] = b.derivative(deriv)(x) if deriv else b(x)
    return np.nan_to_num(out)


def osullivan_penalty(knots_interior, lo, hi, degree=3):
    """Omega_jk = int_lo^hi B_j''(x) B_k''(x) dx, by Gauss-Legendre quadrature per knot interval.

    The null space of Omega is exactly the set of linear functions, whatever the knot spacing
    (O'Sullivan, 1986; Wand and Ormerod, 2008).
    """
    brk = np.unique(np.r_[lo, knots_interior, hi])
    xg, wg = np.polynomial.legendre.leggauss(8)
    xs, ws = [], []
    for a, b in zip(brk[:-1], brk[1:]):
        xs.append((b - a) / 2 * xg + (a + b) / 2)
        ws.append((b - a) / 2 * wg)
    xs, ws = np.concatenate(xs), np.concatenate(ws)
    B2 = bspline_basis(xs, knots_interior, lo, hi, degree, deriv=2)
    return (B2 * ws[:, None]).T @ B2


def diff_penalty(k, order=2):
    dmat = np.diff(np.eye(k), n=order, axis=0)
    return dmat.T @ dmat


# ----------------------------------------------------------------------------- panel algebra
def within_two_way(a, unit, time, tol=1e-10, max_iter=500, trend=None):
    """Two-way demeaning by alternating projections (valid for unbalanced panels).

    With `trend` (a numeric array, e.g. the calendar year) the unit step removes a unit-specific
    linear trend instead of the unit mean, so the result is orthogonal to unit dummies, unit
    dummies x trend and time dummies.
    """
    a = np.asarray(a, float)
    squeeze = a.ndim == 1
    a = a.reshape(len(a), -1).copy()
    u = pd.factorize(unit)[0]
    t = pd.factorize(time)[0]
    nu, nt = u.max() + 1, t.max() + 1
    cu, ct = np.bincount(u, minlength=nu), np.bincount(t, minlength=nt)
    if trend is not None:
        tr = np.asarray(trend, float)
        tr = tr - (np.bincount(u, tr, nu) / cu)[u]          # unit-centred trend
        stt = np.bincount(u, tr * tr, nu)
        stt = np.where(stt > 1e-12, stt, np.inf)
    for _ in range(max_iter):
        old = a.copy()
        for j in range(a.shape[1]):
            a[:, j] -= (np.bincount(u, a[:, j], nu) / cu)[u]
            if trend is not None:
                a[:, j] -= (np.bincount(u, a[:, j] * tr, nu) / stt)[u] * tr
            a[:, j] -= (np.bincount(t, a[:, j], nt) / ct)[t]
        if np.max(np.abs(a - old)) < tol:
            break
    return a[:, 0] if squeeze else a


def mundlak_means(df, unit, cols):
    return df.groupby(unit)[cols].transform("mean").add_suffix("_bar")


def _sup_t(est, cov, n_sim=20000, seed=1, level=0.95):
    """Critical value and p-value of max_j |est_j / sd_j| under N(0, cov)."""
    sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = cov / np.outer(sd, sd)
    w, U = np.linalg.eigh((corr + corr.T) / 2)
    rng = np.random.default_rng(seed)
    sims = np.abs(rng.standard_normal((n_sim, len(est))) @ (U * np.sqrt(np.clip(w, 0, None))).T)
    mx = sims.max(axis=1)
    stat = float(np.max(np.abs(est) / sd))
    return float(np.quantile(mx, level)), stat, float(np.mean(mx >= stat))


# ----------------------------------------------------------------------------- estimator
@dataclass
class PanelDOSE:
    learner_factory: callable
    mode: str = "vc"                 # "vc" (varying coefficient) or "dose" (dose response)
    n_knots: int = 4
    degree: int = 3
    n_folds: int = 5
    n_rep: int = 3
    mundlak: bool = True
    within: bool = True
    penalty: bool = True
    z_in_controls: bool = False      # exploit E[D b(Z)|F] = b(Z) E[D|F] when Z is a control
    mundlak_exclude: tuple = ()      # controls whose unit means are NOT added as features
    select: str = "cv"               # penalty choice: "cv" (leave-units-out) or "gcv"
    cv_folds: int = 10
    lambdas: np.ndarray = field(default_factory=lambda: np.r_[0.0, np.logspace(-6, 4, 41)])
    undersmooth: float = 1.0        # multiply the selected penalty by this factor (<1 undersmooths)
    lambda_fixed: float = None       # if set, use this penalty instead of cross-validation
    pool_lambda: bool = True         # one penalty for all splits, from the CV error summed over splits
    seed: int = 0

    # ---------------------------------------------------------------- sieve
    def _setup_basis(self, s):
        lo, hi = np.quantile(s, [0.005, 0.995])
        qs = np.linspace(0, 1, self.n_knots + 2)[1:-1]
        self.lo_, self.hi_ = lo, hi
        # ties (e.g. a moderator censored at zero) can produce repeated quantiles: keep distinct
        # interior knots only
        kn = np.unique(np.quantile(np.clip(s, lo, hi), qs))
        self.knots_ = kn[(kn > lo) & (kn < hi)]

    def _B(self, s, deriv=0):
        return bspline_basis(s, self.knots_, self.lo_, self.hi_, self.degree, deriv)

    def _phi(self, d, z):
        if self.mode == "vc":
            return d[:, None] * self._B(z)
        # dose: full basis; the level of f is not identified under fixed effects, but every
        # reported functional (f'(d)) is invariant to it and the pseudo-inverse handles the rank.
        return self._B(d)

    def _penalty_matrix(self, K):
        if K <= 2 or self.degree < 2:
            return np.zeros((K, K))
        return osullivan_penalty(self.knots_, self.lo_, self.hi_, self.degree)

    # ---------------------------------------------------------------- fit
    def _w(self, x):
        return within_two_way(x, self.unit_, self.time_, trend=self._trend)

    def fit(self, df, y, d, controls, unit, time, z=None, fold_unit=None, trend=None):
        """fold_unit: column defining the groups for cross-fitting and penalty CV (default: unit).
        In a cluster bootstrap, copies of the same original unit must share a fold, so the
        original identifier is passed here while `unit` holds the relabelled copies.
        trend: numeric column (e.g. the year); if given, the within step also removes
        unit-specific linear trends."""
        df = df.reset_index(drop=True)
        self.unit_, self.time_ = df[unit].values, df[time].values
        self.fold_ = df[fold_unit].values if fold_unit else self.unit_
        self._trend = df[trend].values.astype(float) if trend else None
        Y = df[y].values.astype(float)
        D = df[d].values.astype(float)
        Z = df[z].values.astype(float) if (self.mode == "vc" and z is not None) else D
        self._setup_basis(Z if self.mode == "vc" else D)
        Phi = self._phi(D, Z)
        K = Phi.shape[1]

        feats = df[controls].copy()
        if self.mundlak:
            extra = list(dict.fromkeys([c for c in controls if c not in self.mundlak_exclude]
                                       + [d] + ([z] if z and z not in self.mundlak_exclude
                                                else [])))
            feats = pd.concat([feats, mundlak_means(df, unit, extra)], axis=1)
        tv = df[time]
        feats["_t"] = tv.values if pd.api.types.is_numeric_dtype(tv) else pd.factorize(tv)[0]
        F = feats.values.astype(float)

        self._resid, self._raw = [], []
        rng = np.random.default_rng(self.seed)
        uniq = np.unique(self.fold_)
        for r in range(self.n_rep):
            perm = dict(zip(uniq, rng.permutation(len(uniq))))
            groups = np.array([perm[g] for g in self.fold_])
            ry, rphi = self._crossfit(F, Y, D, Z, Phi, groups)
            self._raw.append((ry.copy(), getattr(self, "_rd", None)))
            if self.within:
                ry = self._w(ry)
                rphi = self._w(rphi)
            self._resid.append((ry, rphi))
        self._Dw = self._w(D) if self.within else D - D.mean()
        self._pen = self._penalty_matrix(K)
        self.K_ = K
        self._Y, self._D, self._Z = Y, D, Z
        self._aggregate()
        return self

    def _aggregate(self):
        pen = self._pen if self.penalty else np.zeros_like(self._pen)
        lam = None
        if self.lambda_fixed is not None:
            lam = float(self.lambda_fixed)
        elif self.pool_lambda and pen.any():
            # a single penalty for all splits: minimise the CV criterion summed over splits, so
            # the reported median is not a mixture of fits with different smoothness
            err = sum(self._cv_error(ry, rphi, pen) for ry, rphi in self._resid)
            lam = float(self.lambdas[int(np.argmin(err))])
        self._splits = [self._final_stage(ry, rphi, pen, lam) for ry, rphi in self._resid]
        betas = np.array([s["b"] for s in self._splits])
        bmed = np.median(betas, axis=0)
        Vs = [s["V"] + np.outer(s["b"] - bmed, s["b"] - bmed) for s in self._splits]
        Vmed = np.median(np.array(Vs), axis=0)
        w, U = np.linalg.eigh((Vmed + Vmed.T) / 2)
        self.cov_ = (U * np.clip(w, 0, None)) @ U.T
        self.coef_ = bmed
        self.lambda_ = float(np.median([s["lam"] for s in self._splits]))
        self.edf_ = float(np.median([s["edf"] for s in self._splits]))

    def refit_final(self, penalty=None, undersmooth=None, lambda_fixed=...):
        """Re-solve the final stage on stored cross-fitted residuals (no new nuisance fits)."""
        if penalty is not None:
            self.penalty = penalty
        if undersmooth is not None:
            self.undersmooth = undersmooth
        if lambda_fixed is not ...:
            self.lambda_fixed = lambda_fixed
        self._aggregate()
        return self

    def rebasis(self, n_knots, degree, penalty=None):
        """Change the sieve without new nuisance fits.

        With Z among the controls, E[D b_k(Z) | F] = b_k(Z) E[D | F], so the orthogonalised
        regressors of *any* basis are b_k(Z) (D - E[D | F]).  Only two nuisance functions are
        ever learned, whatever the sieve dimension.
        """
        assert self.mode == "vc" and self.z_in_controls
        self.n_knots, self.degree = n_knots, degree
        if penalty is not None:
            self.penalty = penalty
        self._setup_basis(self._Z)
        Bz = self._B(self._Z)
        K = Bz.shape[1]
        self._resid = []
        for ry, rd in self._raw:
            rphi = Bz * rd[:, None]
            if self.within:
                ry = self._w(ry)
                rphi = self._w(rphi)
            self._resid.append((ry, rphi))
        self._pen = self._penalty_matrix(K)
        self.K_ = K
        self._aggregate()
        return self

    def _crossfit(self, F, Y, D, Z, Phi, groups):
        n, K = Phi.shape
        ry, rphi, rd = np.empty(n), np.empty((n, K)), np.empty(n)
        for tr, te in GroupKFold(self.n_folds).split(F, groups=groups):
            m = self.learner_factory().fit(F[tr], Y[tr])
            ry[te] = Y[te] - m.predict(F[te])
            if self.mode == "vc" and self.z_in_controls:
                md = self.learner_factory().fit(F[tr], D[tr])
                rd[te] = D[te] - md.predict(F[te])
                rphi[te] = Phi[te] - md.predict(F[te])[:, None] * self._B(Z[te])
            else:
                for k in range(K):
                    mk = self.learner_factory().fit(F[tr], Phi[tr, k])
                    rphi[te, k] = Phi[te, k] - mk.predict(F[te])
        self._rd = rd if (self.mode == "vc" and self.z_in_controls) else None
        return ry, rphi

    # ---------------------------------------------------------------- final stage
    def _scale(self, RtR, pen):
        tp = np.trace(pen)
        return np.trace(RtR) / tp if tp > 0 else 0.0

    def _select_lambda(self, ry, R, pen):
        if not pen.any():
            return 0.0
        return float(self.lambdas[int(np.argmin(self._cv_error(ry, R, pen)))])

    def _cv_error(self, ry, R, pen):
        """CV (or GCV) criterion over the penalty grid for one split."""
        n = len(ry)
        grid = self.lambdas
        RtR = R.T @ R
        sc = self._scale(RtR, pen)
        if self.select == "gcv":
            out = np.empty(len(grid))
            for j, lam in enumerate(grid):
                Ainv = np.linalg.pinv(RtR + lam * sc * pen)
                b = Ainv @ (R.T @ ry)
                edf = np.trace(Ainv @ RtR)
                out[j] = n * np.sum((ry - R @ b) ** 2) / max(n - edf, 1.0) ** 2
            return out
        # leave-units-out cross-validation of the orthogonalised regression (folds formed by
        # the fold units, so bootstrap copies of a country are never split)
        u = pd.factorize(self.fold_)[0]
        folds = GroupKFold(min(self.cv_folds, u.max() + 1)).split(R, groups=u)
        err = np.zeros(len(grid))
        for tr, te in folds:
            Rt, yt = R[tr], ry[tr]
            RtRt, Rty = Rt.T @ Rt, Rt.T @ yt
            sct = self._scale(RtRt, pen)
            for j, lam in enumerate(grid):
                b = np.linalg.pinv(RtRt + lam * sct * pen) @ Rty
                err[j] += np.sum((ry[te] - R[te] @ b) ** 2)
        return err

    def _final_stage(self, ry, R, pen, lam=None):
        n, K = R.shape
        RtR, Rty = R.T @ R, R.T @ ry
        if lam is None:
            lam = self._select_lambda(ry, R, pen)
        lam_used = lam * self.undersmooth
        sc = self._scale(RtR, pen)
        A = RtR + lam_used * sc * pen
        Ainv = np.linalg.pinv(A)
        b = Ainv @ Rty
        e = ry - R @ b
        u = pd.factorize(self.unit_)[0]
        G = u.max() + 1
        S = np.zeros((G, K))
        np.add.at(S, u, R * e[:, None])
        adj = G / (G - 1) * (n - 1) / max(n - K, 1)
        V = adj * Ainv @ (S.T @ S) @ Ainv
        tt = pd.factorize(self.time_)[0]
        St = np.zeros((tt.max() + 1, K))
        np.add.at(St, tt, R * e[:, None])
        return {"b": b, "V": V, "lam": lam, "edf": float(np.trace(Ainv @ RtR)),
                "infl": S @ Ainv.T, "infl_t": St @ Ainv.T, "infl_h": (R * e[:, None]) @ Ainv.T,
                "adj": adj}

    # ---------------------------------------------------------------- functionals
    def _functional(self, L):
        """Median-aggregated estimate and covariance of the vector of functionals L @ beta.

        Each split s gives c_s = L b_s and C_s = L V_s L'.  Following Chernozhukov et al. (2018),
        the estimate is the element-wise median of c_s and the covariance the element-wise median
        of C_s + (c_s - c)(c_s - c)', projected on the positive semi-definite cone.  Any linear
        transformation of the functionals (centring, detrending, contrasts) is applied to L
        *before* aggregation, so every reported quantity is aggregated functional by functional.
        """
        L = np.atleast_2d(L)
        ests = np.array([L @ s["b"] for s in self._splits])
        med = np.median(ests, axis=0)
        Cs = np.array([L @ s["V"] @ L.T + np.outer(e - med, e - med)
                       for s, e in zip(self._splits, ests)])
        C = np.median(Cs, axis=0)
        w, U = np.linalg.eigh((C + C.T) / 2)
        C = (U * np.clip(w, 0, None)) @ U.T
        # keep the median variances exactly on the diagonal
        var = np.median(Cs[:, np.arange(len(med)), np.arange(len(med))], axis=0)
        d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
        sd = np.sqrt(np.clip(var, 0, None))
        C = C / np.outer(d, d) * np.outer(sd, sd)
        return med, sd, C

    def _L_theta(self, grid):
        return self._B(grid) if self.mode == "vc" else self._B(grid, deriv=1)

    def effect(self, grid, level=0.95, n_sim=20000, seed=1):
        """theta(z) (vc) or marginal effect f'(d) (dose) on a grid with pointwise and sup-t bands."""
        grid = np.asarray(grid, float)
        est, se, C = self._functional(self._L_theta(grid))
        zc = stats.norm.ppf(0.5 + level / 2)
        crit, _, _ = _sup_t(est, C, n_sim, seed, level)
        return pd.DataFrame({"grid": grid, "est": est, "se": se,
                             "lo": est - zc * se, "hi": est + zc * se,
                             "ulo": est - crit * se, "uhi": est + crit * se})

    def _L_mean(self, mask=None):
        s = self._Z if self.mode == "vc" else self._D
        if mask is not None:
            s = s[mask]
        return (self._B(s) if self.mode == "vc" else self._B(s, deriv=1)).mean(axis=0)

    def average_effect(self, weights=None):
        """Sample-average effect: mean of theta(Z_it) (vc) or of f'(D_it) (dose)."""
        if weights is None:
            l = self._L_mean()
        else:
            w = np.asarray(weights, float) / np.sum(weights)
            s = self._Z if self.mode == "vc" else self._D
            B = self._B(s) if self.mode == "vc" else self._B(s, deriv=1)
            l = (B * w[:, None]).sum(axis=0)
        m, s, _ = self._functional(l)
        return float(m[0]), float(s[0])

    def group_effect(self, mask):
        """Average of theta(Z_it) over the observations selected by a boolean mask."""
        m, s, _ = self._functional(self._L_mean(mask))
        return float(m[0]), float(s[0])

    def group_contrast(self, mask_a, mask_b):
        """Difference of group-average effects, theta-bar(A) - theta-bar(B), with its s.e."""
        m, s, _ = self._functional(self._L_mean(mask_a) - self._L_mean(mask_b))
        return float(m[0]), float(s[0])

    def blp_slope(self):
        """Best linear projection slope of theta(Z) on Z (unweighted, as defined in the paper):
        gamma = Cov(theta(Z), Z) / Var(Z) over the estimation sample."""
        assert self.mode == "vc"
        zc = self._Z - self._Z.mean()
        l = (self._B(self._Z) * zc[:, None]).mean(axis=0) / np.mean(zc ** 2)
        m, s, _ = self._functional(l)
        return float(m[0]), float(s[0])

    def shape_tests(self, n_grid=25, n_sim=20000, seed=3):
        """Sup-t tests that theta(.) is constant and that it is linear on a quantile grid.

        The statistic is max_j |c_j| / sd(c_j) for the centred (constancy) or linearly detrended
        (linearity) values c = M L beta.  The transformation M is applied to each split's
        coefficients before the median aggregation, and the critical distribution is simulated
        from the aggregated (possibly rank-deficient) covariance of c, so no matrix inversion is
        needed.  Constant and linear functions lie in the null space of the O'Sullivan penalty,
        so under either null hypothesis the penalised estimator of c has no smoothing bias for
        a given penalty.
        """
        s = self._Z if self.mode == "vc" else self._D
        grid = np.quantile(s, np.linspace(0.05, 0.95, n_grid))
        Lg = self._L_theta(grid)
        out = {}
        for name, X in [("const", np.ones((n_grid, 1))),
                        ("lin", np.c_[np.ones(n_grid), grid])]:
            M = np.eye(n_grid) - X @ np.linalg.pinv(X)
            c, sd, Cc = self._functional(M @ Lg)
            keep = sd > 1e-6 * np.max(sd)
            _, stat, p = _sup_t(c[keep], Cc[np.ix_(keep, keep)], n_sim, seed)
            out[name] = (stat, p)
        return out

    def influence(self, l):
        """First-order per-unit influence contributions of l'beta, averaged over splits.

        A diagnostic: it holds the nuisance fits, folds, basis and penalty fixed, so it is a
        linear approximation to the effect of deleting a unit, not a refit.
        """
        u_codes, u_names = pd.factorize(self.unit_)
        phis = np.mean([s["infl"] @ l for s in self._splits], axis=0)
        adj = float(np.median([s["adj"] for s in self._splits]))
        return pd.Series(phis, index=u_names), adj

    def twoway_se(self, l):
        """Standard error of l'beta clustered by unit and by time (Cameron, Gelbach and Miller,
        2011): V = c_G V_unit + c_T V_time - c_n V_obs, with c_G = G/(G-1), c_T = T/(T-1) and
        c_n = n/(n-K).  If the combination is negative (possible in finite samples) the unit-
        clustered variance (with the same small-sample factor as V) is used.  Split aggregation
        as in _functional."""
        ests, vs = [], []
        n = len(self.unit_)
        G = len(np.unique(self.unit_))
        T = len(np.unique(self.time_))
        for s in self._splits:
            vu = np.sum((s["infl"] @ l) ** 2) * G / (G - 1)
            vt = np.sum((s["infl_t"] @ l) ** 2) * T / (T - 1)
            vh = np.sum((s["infl_h"] @ l) ** 2) * n / max(n - self.K_, 1)
            v = vu + vt - vh
            vs.append(v if v > 0 else float(l @ s["V"] @ l))
            ests.append(float(s["b"] @ l))
        ests = np.array(ests)
        med = np.median(ests)
        return float(np.sqrt(np.median(np.array(vs) + (ests - med) ** 2)))

    def identifying_variation(self, masks=None):
        """Share of the within-transformed treatment variance left after the nuisance step."""
        rd = [r[1] for r in self._raw] if self.z_in_controls else None
        if rd is None or rd[0] is None:
            return None
        rdw = np.mean([self._w(x) for x in rd], axis=0)
        out = {"all": float(np.var(rdw) / np.var(self._Dw)),
               "all_total": float(np.var(rdw) / np.var(self._D))}
        for k, m in (masks or {}).items():
            out[k] = float(np.var(rdw[m]) / np.var(self._Dw[m]))
        return out
