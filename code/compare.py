"""Compare the heterogeneity estimates of two data sources on their common country-years.

Two comparisons, both along initial productivity (high-minus-low tercile contrast):

* ``pwt``: Penn World Table 10.0 (a) vs 11.0 (b), common country-years 1996-2019;
* ``wdi``: World Bank WDI (a) vs PWT 11.0 (b), common country-years 1996-2023.

For each comparison:

1. component replacement: the outcome (y), the moderator (z) and the control set (x) are taken
   from either source (2^3 = 8 configurations), each fully refitted, with an order-free Shapley
   attribution of the total change;
2. the same with a fixed smoother: moderator as within-source percentile rank (same knots),
   tercile groups of source b, and one penalty for every configuration;
3. the split-seed distribution of the two all-a / all-b contrasts;
4. a paired country bootstrap of the difference: both sources refitted on the same resampled
   countries (nuisance functions, penalty and terciles re-estimated), with copies of a country
   kept in the same cross-fitting and penalty-CV fold;
5. leave-one-country-out refits of both sources with the penalty, the split seed and the tercile
   cutoffs held at their full-sample values.

Output: results/compare_<kind>.json, compare_<kind>_bootstrap.csv, compare_<kind>_loco.csv.
Run as ``python code/compare.py <kind> <n_jobs> <n_boot>``.
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from empirical import BASE, N_MAIN, N_ROB, PWT_BASE, fit_dose  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PROC = ROOT / "data" / "processed"
KIND = sys.argv[1] if len(sys.argv) > 1 else "pwt"
N_JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N_BOOT = int(sys.argv[3]) if len(sys.argv) > 3 else 199
N_SEEDS = 10


def common_sample(kind=KIND):
    """Merged panel with source-a columns suffixed _a and source-b columns suffixed _b."""
    keys = ["iso", "year"]
    if kind == "pwt":
        a = pd.read_csv(PROC / "panel_pwt.csv")
        b = pd.read_csv(PROC / "panel_pwt11.csv")
        xa = xb = [x for x in PWT_BASE if x != "lp_init"]
    else:
        a = pd.read_csv(PROC / "panel.csv")
        a = a[a["year"] <= 2023]
        b = pd.read_csv(PROC / "panel_pwt11.csv")
        xa = [x for x in BASE if x != "lp_init"]
        xb = [x for x in PWT_BASE if x != "lp_init"]
    ca = ["dlp", "lp_init", "internet_l1"] + xa
    cb = ["dlp", "lp_init"] + xb
    c = a[keys + ca].add_suffix("_a").rename(columns={"iso_a": "iso", "year_a": "year"}).merge(
        b[keys + cb].add_suffix("_b").rename(columns={"iso_b": "iso", "year_b": "year"}),
        on=keys).rename(columns={"internet_l1_a": "internet_l1"})
    c = c.dropna().reset_index(drop=True)
    for v in ["a", "b"]:
        c[f"rank_{v}"] = c[f"lp_init_{v}"].rank(pct=True)
    c["iso0"] = c["iso"]
    return c, [f"{x}_a" for x in xa], [f"{x}_b" for x in xb]


def contrast(c, vy, vz, vx, xs, fixed=None, cut=None, n_rep=N_MAIN, seed=11, lam=None):
    """Tercile contrast for one configuration (vy, vz, vx in {'a', 'b'})."""
    ctrl = xs[vx]
    kw = {"n_rep": n_rep, "seed": seed, "fold_unit": "iso0"}
    if fixed is None:
        z = f"lp_init_{vz}"
        if lam is not None:
            kw["lambda_fixed"] = lam
        m = fit_dose(c, y=f"dlp_{vy}", z=z, controls=ctrl, **kw)
        zz = m._Z
        q1, q2 = np.quantile(zz, [1 / 3, 2 / 3]) if cut is None else cut
        lo, hi = zz <= q1, zz > q2
    else:
        m = fit_dose(c, y=f"dlp_{vy}", z=f"rank_{vz}", controls=ctrl, lambda_fixed=fixed["lam"],
                     **kw)
        lo, hi = fixed["groups"]
    d, dse = m.group_contrast(hi, lo)
    gl, gls = m.group_effect(lo)
    return {"diff": d, "diff_se": dse, "gate_low": gl, "gate_low_se": gls, "lambda": m.lambda_,
            "edf": m.edf_}


def shapley(v):
    """Shapley attribution of v[(1,1,1)] - v[(0,0,0)] to the three components."""
    out = []
    for k in range(3):
        tot = 0.0
        for S in itertools.product([0, 1], repeat=3):
            if S[k] == 1:
                continue
            with_k = list(S)
            with_k[k] = 1
            s = sum(S)
            w = math.factorial(s) * math.factorial(3 - s - 1) / math.factorial(3)
            tot += w * (v[tuple(with_k)] - v[S])
        out.append(tot)
    return out


def boot_draw(c, xs, b):
    rng = np.random.default_rng(5000 + b)
    iso = c["iso0"].unique()
    draw = rng.choice(iso, len(iso), replace=True)
    # copies get distinct unit labels (fixed effects, clustering) but keep the original label
    # in iso0, which defines the cross-fitting and penalty-CV folds
    parts = [c[c["iso0"] == g].assign(iso=f"{g}_{k}") for k, g in enumerate(draw)]
    cb = pd.concat(parts, ignore_index=True)
    ra = contrast(cb, "a", "a", "a", xs, n_rep=N_ROB)
    rb = contrast(cb, "b", "b", "b", xs, n_rep=N_ROB)
    return {"diff_a": ra["diff"], "diff_b": rb["diff"], "low_a": ra["gate_low"],
            "low_b": rb["gate_low"], "lam_a": ra["lambda"], "lam_b": rb["lambda"]}


def loco(c, xs, g, cuts, lams):
    cc = c[c["iso0"] != g]
    ra = contrast(cc, "a", "a", "a", xs, cut=cuts["a"], n_rep=N_ROB, lam=lams["a"])
    rb = contrast(cc, "b", "b", "b", xs, cut=cuts["b"], n_rep=N_ROB, lam=lams["b"])
    return {"iso": g, "diff_a": ra["diff"], "diff_b": rb["diff"], "low_a": ra["gate_low"],
            "low_b": rb["gate_low"]}


def main():
    c, xa, xb = common_sample()
    xs = {"a": xa, "b": xb}
    configs = list(itertools.product([0, 1], repeat=3))   # (y, z, x): 0 = source a, 1 = b
    lab = lambda t: tuple("b" if v else "a" for v in t)   # noqa: E731

    full = Parallel(n_jobs=N_JOBS)(delayed(contrast)(c, *lab(t), xs) for t in configs)
    full = dict(zip(configs, full))
    r = c["rank_b"].values
    groups = (r <= 1 / 3, r > 2 / 3)
    lam_fx = fit_dose(c, y="dlp_b", z="rank_b", controls=xb, fold_unit="iso0").lambda_
    fixed = Parallel(n_jobs=N_JOBS)(delayed(contrast)(c, *lab(t), xs,
                                                      fixed={"lam": lam_fx, "groups": groups})
                                    for t in configs)
    fixed = dict(zip(configs, fixed))
    print("configurations done", flush=True)

    out = {"kind": KIND, "n_common": len(c), "n_countries": int(c["iso"].nunique()),
           "years": [int(c["year"].min()), int(c["year"].max())], "fixed_lambda": lam_fx,
           "configs": [{"y": lab(t)[0], "z": lab(t)[1], "x": lab(t)[2], **full[t],
                        **{f"fixed_{k}": v for k, v in fixed[t].items()}} for t in configs]}
    for name, res in [("full", full), ("fixed", fixed)]:
        v = {t: res[t]["diff"] for t in configs}
        out[f"shapley_{name}"] = dict(zip(["outcome", "moderator", "controls"], shapley(v)))
        out[f"total_{name}"] = v[(1, 1, 1)] - v[(0, 0, 0)]
    ta = pd.qcut(c["lp_init_a"], 3, labels=False)
    tb = pd.qcut(c["lp_init_b"], 3, labels=False)
    out["tercile_switch_share"] = float(np.mean(ta != tb))
    out["countries_switching"] = int(c.loc[ta != tb, "iso"].nunique())
    out["rank_corr"] = float(np.corrcoef(c["rank_a"], c["rank_b"])[0, 1])
    out["growth_corr_annual"] = float(np.corrcoef(c["dlp_a"], c["dlp_b"])[0, 1])

    # split-seed distribution
    seeds = Parallel(n_jobs=N_JOBS)(delayed(contrast)(c, v, v, v, xs, seed=s)
                                    for v in ["a", "b"] for s in range(100, 100 + N_SEEDS))
    da = [x["diff"] for x in seeds[:N_SEEDS]]
    db = [x["diff"] for x in seeds[N_SEEDS:]]
    out["seeds"] = {"diff_a": da, "diff_b": db, "diff": list(np.array(da) - np.array(db))}
    print("seeds done", flush=True)

    bs = pd.DataFrame(Parallel(n_jobs=N_JOBS, verbose=2)(delayed(boot_draw)(c, xs, b)
                                                          for b in range(N_BOOT)))
    bs.to_csv(RES / f"compare_{KIND}_bootstrap.csv", index=False)
    a0, b1 = full[(0, 0, 0)], full[(1, 1, 1)]
    est = a0["diff"] - b1["diff"]
    dd = bs["diff_a"] - bs["diff_b"]
    se = float(dd.std(ddof=1))
    out["difference"] = {
        "est": est, "boot_se": se, "boot_mean": float(dd.mean()),
        "boot_bias": float(dd.mean() - est),
        "boot_ci_percentile": [float(dd.quantile(0.025)), float(dd.quantile(0.975))],
        "boot_ci_normal": [est - 1.96 * se, est + 1.96 * se],
        # basic (reverse-percentile) interval, which corrects for bootstrap bias
        "boot_ci_basic": [float(2 * est - dd.quantile(0.975)), float(2 * est - dd.quantile(0.025))],
        "n_boot": int(len(bs))}
    for v, r_ in [("a", a0), ("b", b1)]:
        x = bs[f"diff_{v}"]
        out[f"contrast_{v}"] = {"est": r_["diff"], "se": r_["diff_se"], "boot_se": float(x.std(ddof=1)),
                                "boot_mean": float(x.mean()),
                                "boot_ci_basic": [float(2 * r_["diff"] - x.quantile(0.975)),
                                                  float(2 * r_["diff"] - x.quantile(0.025))],
                                "boot_share_le0": float(np.mean(x <= 0))}
    print("bootstrap done", flush=True)

    za, zb = c["lp_init_a"].values, c["lp_init_b"].values
    cuts = {"a": tuple(np.quantile(za, [1 / 3, 2 / 3])), "b": tuple(np.quantile(zb, [1 / 3, 2 / 3]))}
    lams = {"a": a0["lambda"], "b": b1["lambda"]}
    lo = pd.DataFrame(Parallel(n_jobs=N_JOBS, verbose=2)(delayed(loco)(c, xs, g, cuts, lams)
                                                          for g in c["iso0"].unique()))
    lo.to_csv(RES / f"compare_{KIND}_loco.csv", index=False)
    dlo = lo["diff_a"] - lo["diff_b"]
    i = (dlo - est).abs().idxmax()
    out["loco"] = {"range": [float(dlo.min()), float(dlo.max())], "median": float(dlo.median()),
                   "max_iso": lo.loc[i, "iso"], "max_change": float(dlo[i] - est),
                   "range_a": [float(lo["diff_a"].min()), float(lo["diff_a"].max())],
                   "range_b": [float(lo["diff_b"].min()), float(lo["diff_b"].max())]}
    json.dump(out, open(RES / f"compare_{KIND}.json", "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in out if k != "configs"}, indent=1, default=float))


if __name__ == "__main__":
    main()
