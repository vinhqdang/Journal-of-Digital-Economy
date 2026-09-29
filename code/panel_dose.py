"""Panel-DOSE: a Debiased Orthogonal Sieve Estimator for heterogeneous effects in panels.

Target model
------------
    Y_it = theta(Z_it) * D_it + g(X_it) + alpha_i + lambda_t + e_it        (varying-coefficient mode)
    Y_it = f(D_it)            + g(X_it) + alpha_i + lambda_t + e_it        (dose-response mode)

theta(.) or f(.) is approximated by a penalised cubic B-spline sieve, phi(D, Z)'beta, while
g(.) is left unrestricted and learned with machine learning.  The algorithm

1. augments the controls with Mundlak (within-unit) means so that the nuisance learners can
   absorb the part of the fixed effect that is correlated with observables;
2. cross-fits the nuisance functions E[Y | F] and E[phi_k | F] with folds formed by *units*,
   so no country is used to predict itself;
3. two-way demeans the cross-fitted residuals to remove any remaining additive unit and time
   effects;
4. solves the Neyman-orthogonal moment for beta with a second-difference (P-spline) penalty
   whose strength is chosen by generalised cross-validation;
5. repeats the sample split and aggregates by the median, with a cluster-robust sandwich
   variance and sup-t simultaneous bands for theta(.).
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


def diff_penalty(k, order=2):
    dmat = np.diff(np.eye(k), n=order, axis=0)
    return dmat.T @ dmat


# ----------------------------------------------------------------------------- panel algebra
def within_two_way(a, unit, time, tol=1e-10, max_iter=500):
    """Two-way demeaning by alternating projections (valid for unbalanced panels)."""
    a = np.asarray(a, float)
    squeeze = a.ndim == 1
    a = a.reshape(len(a), -1).copy()
    u = pd.factorize(unit)[0]
    t = pd.factorize(time)[0]
    nu, nt = u.max() + 1, t.max() + 1
    cu, ct = np.bincount(u, minlength=nu), np.bincount(t, minlength=nt)
    for _ in range(max_iter):
        old = a.copy()
        for j in range(a.shape[1]):
            a[:, j] -= (np.bincount(u, a[:, j], nu) / cu)[u]
            a[:, j] -= (np.bincount(t, a[:, j], nt) / ct)[t]
        if np.max(np.abs(a - old)) < tol:
            break
    return a[:, 0] if squeeze else a


def mundlak_means(df, unit, cols):
    return df.groupby(unit)[cols].transform("mean").add_suffix("_bar")


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
    lambdas: np.ndarray = field(default_factory=lambda: np.r_[0.0, np.logspace(-4, 3, 36)])
    undersmooth: float = 1.0        # multiply the GCV penalty by this factor (<1 undersmooths)
    seed: int = 0

    # ---------------------------------------------------------------- sieve
    def _setup_basis(self, s):
        lo, hi = np.quantile(s, [0.005, 0.995])
        qs = np.linspace(0, 1, self.n_knots + 2)[1:-1]
        self.lo_, self.hi_ = lo, hi
        self.knots_ = np.quantile(np.clip(s, lo, hi), qs)

    def _B(self, s, deriv=0):
        return bspline_basis(s, self.knots_, self.lo_, self.hi_, self.degree, deriv)

    def _phi(self, d, z):
        if self.mode == "vc":
            return d[:, None] * self._B(z)
        # dose: drop first column (level not identified under fixed effects)
        return self._B(d)[:, 1:]

    # ---------------------------------------------------------------- fit
    def fit(self, df, y, d, controls, unit, time, z=None):
        df = df.reset_index(drop=True)
        self.unit_, self.time_ = df[unit].values, df[time].values
        Y = df[y].values.astype(float)
        D = df[d].values.astype(float)
        Z = df[z].values.astype(float) if (self.mode == "vc" and z is not None) else D
        self._setup_basis(Z if self.mode == "vc" else D)
        Phi = self._phi(D, Z)
        K = Phi.shape[1]

        feats = df[controls].copy()
        if self.mundlak:
            extra = list(dict.fromkeys(controls + [d] + ([z] if z else [])))
            feats = pd.concat([feats, mundlak_means(df, unit, extra)], axis=1)
        feats["_t"] = df[time].values
        F = feats.values.astype(float)

        pen = diff_penalty(K, 2) if K > 2 else np.zeros((K, K))
        if self.mode == "dose" and K > 2:
            pen = diff_penalty(K + 1, 2)[1:, 1:]

        self._resid, self._raw = [], []
        rng = np.random.default_rng(self.seed)
        uniq = np.unique(self.unit_)
        for r in range(self.n_rep):
            perm = dict(zip(uniq, rng.permutation(len(uniq))))
            groups = np.array([perm[g] for g in self.unit_])
            ry, rphi = self._crossfit(F, Y, D, Z, Phi, groups)
            self._raw.append((ry.copy(), getattr(self, "_rd", None)))
            if self.within:
                ry = within_two_way(ry, self.unit_, self.time_)
                rphi = within_two_way(rphi, self.unit_, self.time_)
            self._resid.append((ry, rphi))
        self._pen = pen
        self._aggregate()
        self.K_ = K
        self._Y, self._D, self._Z = Y, D, Z
        return self

    def _aggregate(self):
        pen = self._pen if self.penalty else np.zeros_like(self._pen)
        betas, covs, lams = [], [], []
        for ry, rphi in self._resid:
            b, V, lam = self._final_stage(ry, rphi, pen)
            betas.append(b), covs.append(V), lams.append(lam)
        betas = np.array(betas)
        bmed = np.median(betas, axis=0)
        Vs = [V + np.outer(b - bmed, b - bmed) for b, V in zip(betas, covs)]
        # element-wise median of the adjusted covariances, projected back to PSD
        Vmed = np.median(np.array(Vs), axis=0)
        w, U = np.linalg.eigh((Vmed + Vmed.T) / 2)
        self.cov_ = (U * np.clip(w, 0, None)) @ U.T
        self.coef_, self.lambda_ = bmed, float(np.median(lams))

    def refit_final(self, penalty=None, undersmooth=None):
        """Re-solve the final stage on stored cross-fitted residuals (no new nuisance fits)."""
        if penalty is not None:
            self.penalty = penalty
        if undersmooth is not None:
            self.undersmooth = undersmooth
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
                ry = within_two_way(ry, self.unit_, self.time_)
                rphi = within_two_way(rphi, self.unit_, self.time_)
            self._resid.append((ry, rphi))
        self._pen = diff_penalty(K, 2) if K > 2 else np.zeros((K, K))
        self.K_ = K
        self._aggregate()
        return self

    def group_effect(self, mask):
        """Average of theta(Z_it) over the observations selected by a boolean mask."""
        l = self._B(self._Z[mask]).mean(axis=0)
        return float(l @ self.coef_), float(np.sqrt(l @ self.cov_ @ l))

    def group_contrast(self, mask_a, mask_b):
        """Difference of group-average effects, theta-bar(A) - theta-bar(B), with its s.e."""
        l = self._B(self._Z[mask_a]).mean(axis=0) - self._B(self._Z[mask_b]).mean(axis=0)
        return float(l @ self.coef_), float(np.sqrt(l @ self.cov_ @ l))

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

    def _final_stage(self, ry, R, pen):
        n, K = R.shape
        RtR, Rty = R.T @ R, R.T @ ry
        best = (np.inf, None, None)
        for lam in (self.lambdas if pen.any() else [0.0]):
            A = RtR + lam * n * pen
            Ainv = np.linalg.pinv(A)
            b = Ainv @ Rty
            edf = np.trace(Ainv @ RtR)
            rss = np.sum((ry - R @ b) ** 2)
            gcv = n * rss / max(n - edf, 1.0) ** 2
            if gcv < best[0]:
                best = (gcv, lam, b)
        _, lam, b = best
        lam *= self.undersmooth
        A = RtR + lam * n * pen
        Ainv = np.linalg.pinv(A)
        b = Ainv @ Rty
        e = ry - R @ b
        # cluster-robust meat
        u = pd.factorize(self.unit_)[0]
        S = np.zeros((u.max() + 1, K))
        np.add.at(S, u, R * e[:, None])
        G = u.max() + 1
        adj = G / (G - 1) * (n - 1) / max(n - K, 1)
        V = adj * Ainv @ (S.T @ S) @ Ainv
        return b, V, lam

    # ---------------------------------------------------------------- inference
    def effect(self, grid, level=0.95, n_sim=20000, seed=1):
        """theta(z) (vc) or marginal effect f'(d) (dose) on a grid with pointwise and sup-t bands."""
        grid = np.asarray(grid, float)
        if self.mode == "vc":
            L = self._B(grid)
        else:
            L = self._B(grid, deriv=1)[:, 1:]
        est = L @ self.coef_
        cov = L @ self.cov_ @ L.T
        se = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        zc = stats.norm.ppf(0.5 + level / 2)
        corr = cov / np.outer(se, se)
        rng = np.random.default_rng(seed)
        w, U = np.linalg.eigh((corr + corr.T) / 2)
        sims = rng.standard_normal((n_sim, len(grid))) @ (U * np.sqrt(np.clip(w, 0, None))).T
        crit = np.quantile(np.abs(sims).max(axis=1), level)
        return pd.DataFrame({"grid": grid, "est": est, "se": se,
                             "lo": est - zc * se, "hi": est + zc * se,
                             "ulo": est - crit * se, "uhi": est + crit * se})

    def average_effect(self):
        """Sample-average effect: mean of theta(Z_it) (vc) or of f'(D_it) (dose)."""
        L = self._B(self._Z) if self.mode == "vc" else self._B(self._D, deriv=1)[:, 1:]
        l = L.mean(axis=0)
        return float(l @ self.coef_), float(np.sqrt(l @ self.cov_ @ l))

    def wald(self, order=1):
        """Wald test that the sieve coefficients' differences of given order vanish.

        order=1: theta(.) constant (vc) / f(.) linear (dose)
        order=2: theta(.) linear in z (vc)
        """
        K = self.K_ if self.mode == "vc" else self.K_ + 1
        C = np.diff(np.eye(K), n=order, axis=0)
        if self.mode == "dose":
            C = C[:, 1:]
        cb = C @ self.coef_
        V = C @ self.cov_ @ C.T
        stat = float(cb @ np.linalg.pinv(V) @ cb)
        df = np.linalg.matrix_rank(V)
        return stat, int(df), float(stats.chi2.sf(stat, df))
