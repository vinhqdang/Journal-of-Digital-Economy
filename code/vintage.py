"""Comparison of Penn World Table releases 10.0 and 11.0 on their common country-years.

For the high-minus-low tercile contrast along initial productivity:

1. component replacement: the outcome (y), the moderator (z) and the controls (x) are taken from
   either release, giving 2^3 = 8 configurations, each fully refitted (nuisance, penalty,
   terciles), and an order-free Shapley attribution of the total change to the three components;
2. the same exercise with the tercile groups, the knots (moderator as a within-release
   percentile rank) and the penalty held fixed, so that every configuration estimates the same
   population functional with the same smoother;
3. a paired country bootstrap of the full pipeline (both releases refitted on the same resampled
   countries; nuisance, penalty and terciles re-estimated in every draw) for the standard error
   of the difference between the releases;
4. leave-one-country-out refits of both releases.

Output: results/vintage.json.  Run as `python code/vintage.py <n_jobs> <n_boot>`.
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
from empirical import PWT_BASE, fit_dose  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PROC = ROOT / "data" / "processed"
XS = [x for x in PWT_BASE if x != "lp_init"]
N_JOBS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 199


def common_sample():
    p10 = pd.read_csv(PROC / "panel_pwt.csv")
    p11 = pd.read_csv(PROC / "panel_pwt11.csv")
    keys = ["iso", "year"]
    cols = ["dlp", "lp_init", "internet_l1"] + XS
    c = p10[keys + cols].merge(p11[keys + cols], on=keys, suffixes=("_10", "_11"))
    c = c.rename(columns={"internet_l1_10": "internet_l1"}).drop(columns=["internet_l1_11"])
    for v in ["10", "11"]:
        c[f"rank_{v}"] = c[f"lp_init_{v}"].rank(pct=True)
    return c.reset_index(drop=True)


def contrast(c, vy, vz, vx, fixed=None, cut=None):
    """Tercile contrast and low-tercile average for one configuration.

    fixed = None: full refit (terciles of the moderator used).  fixed = dict(lam=..., groups=...)
    uses the rank-transformed moderator, the given penalty and the given tercile groups.
    cut = (q1, q2): tercile cutoffs of the moderator (default: computed on the data).
    """
    if fixed is None:
        z = f"lp_init_{vz}"
        m = fit_dose(c, y=f"dlp_{vy}", z=z, controls=[f"{x}_{vx}" for x in XS])
        zz = m._Z
        q1, q2 = np.quantile(zz, [1 / 3, 2 / 3]) if cut is None else cut
        lo, hi = zz <= q1, zz > q2
    else:
        # rank moderator: the tercile groups and the knots are the same in every configuration
        z = f"rank_{vz}"
        m = fit_dose(c, y=f"dlp_{vy}", z=z, controls=[f"{x}_{vx}" for x in XS],
                     lambda_fixed=fixed["lam"])
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


def boot_draw(c, b):
    rng = np.random.default_rng(5000 + b)
    iso = c["iso"].unique()
    draw = rng.choice(iso, len(iso), replace=True)
    parts = [c[c["iso"] == g].assign(iso=f"{g}_{k}") for k, g in enumerate(draw)]
    cb = pd.concat(parts, ignore_index=True)
    r10 = contrast(cb, "10", "10", "10")
    r11 = contrast(cb, "11", "11", "11")
    return {"diff10": r10["diff"], "diff11": r11["diff"], "low10": r10["gate_low"],
            "low11": r11["gate_low"]}


def loco(c, g, cut10, cut11):
    cc = c[c["iso"] != g]
    r10 = contrast(cc, "10", "10", "10", cut=cut10)
    r11 = contrast(cc, "11", "11", "11", cut=cut11)
    return {"iso": g, "diff10": r10["diff"], "diff11": r11["diff"], "low10": r10["gate_low"],
            "low11": r11["gate_low"]}


def main():
    c = common_sample()
    configs = list(itertools.product([0, 1], repeat=3))  # (y, z, x): 0 = PWT 10.0, 1 = PWT 11.0
    lab = lambda t: tuple("11" if a else "10" for a in t)  # noqa: E731

    # 1. full refits
    full = Parallel(n_jobs=N_JOBS)(delayed(contrast)(c, *lab(t)) for t in configs)
    full = dict(zip(configs, full))
    print("full refits done", flush=True)
    # 2. fixed groups, knots and penalty (rank moderator; groups and lambda from PWT 11.0)
    r = c["rank_11"].values
    groups = (r <= 1 / 3, r > 2 / 3)
    lam = fit_dose(c, y="dlp_11", z="rank_11", controls=[f"{x}_11" for x in XS]).lambda_
    fixed = Parallel(n_jobs=N_JOBS)(delayed(contrast)(c, *lab(t), fixed={"lam": lam,
                                                                          "groups": groups})
                                    for t in configs)
    fixed = dict(zip(configs, fixed))
    print("fixed refits done", flush=True)

    out = {"n_common": len(c), "n_countries": int(c["iso"].nunique()),
           "configs": [{"y": lab(t)[0], "z": lab(t)[1], "x": lab(t)[2], **full[t],
                        **{f"fixed_{k}": v for k, v in fixed[t].items()}} for t in configs],
           "fixed_lambda": lam}
    for name, res in [("full", full), ("fixed", fixed)]:
        v = {t: res[t]["diff"] for t in configs}
        out[f"shapley_{name}"] = dict(zip(["outcome", "moderator", "controls"], shapley(v)))
        out[f"total_{name}"] = v[(1, 1, 1)] - v[(0, 0, 0)]
    t10 = pd.qcut(c["lp_init_10"], 3, labels=False)
    t11 = pd.qcut(c["lp_init_11"], 3, labels=False)
    out["tercile_switch_share"] = float(np.mean(t10 != t11))
    out["countries_switching"] = int(c.loc[t10 != t11, "iso"].nunique())
    out["rank_corr"] = float(np.corrcoef(c["rank_10"], c["rank_11"])[0, 1])
    g10 = c.groupby("iso")["dlp_10"].mean()
    g11 = c.groupby("iso")["dlp_11"].mean()
    out["growth_corr_country_means"] = float(np.corrcoef(g10, g11)[0, 1])
    out["growth_corr_annual"] = float(np.corrcoef(c["dlp_10"], c["dlp_11"])[0, 1])

    # 3. paired country bootstrap of the full pipeline
    bs = pd.DataFrame(Parallel(n_jobs=N_JOBS, verbose=2)(delayed(boot_draw)(c, b)
                                                          for b in range(N_BOOT)))
    bs.to_csv(RES / "vintage_bootstrap.csv", index=False)
    est = full[(0, 0, 0)]["diff"] - full[(1, 1, 1)]["diff"]
    dd = bs["diff10"] - bs["diff11"]
    out["difference"] = {"est": est, "boot_se": float(dd.std(ddof=1)),
                         "boot_mean": float(dd.mean()),
                         "boot_ci": [float(dd.quantile(0.025)), float(dd.quantile(0.975))],
                         "boot_p_le0": float(np.mean(dd <= 0)), "n_boot": int(len(bs)),
                         "diff10_boot_se": float(bs["diff10"].std(ddof=1)),
                         "diff11_boot_se": float(bs["diff11"].std(ddof=1)),
                         "diff10_boot_mean": float(bs["diff10"].mean()),
                         "diff11_boot_mean": float(bs["diff11"].mean()),
                         "low10_boot_se": float(bs["low10"].std(ddof=1))}
    print("bootstrap done", flush=True)

    # 4. leave-one-country-out refits (tercile cutoffs of the full sample kept fixed)
    z10, z11 = c["lp_init_10"].values, c["lp_init_11"].values
    cut10 = tuple(np.quantile(z10, [1 / 3, 2 / 3]))
    cut11 = tuple(np.quantile(z11, [1 / 3, 2 / 3]))
    lo = pd.DataFrame(Parallel(n_jobs=N_JOBS, verbose=2)(delayed(loco)(c, g, cut10, cut11)
                                                          for g in c["iso"].unique()))
    lo.to_csv(RES / "vintage_loco.csv", index=False)
    b10, b11 = full[(0, 0, 0)], full[(1, 1, 1)]
    lo["ch_diff10"] = lo["diff10"] - b10["diff"]
    lo["ch_diff11"] = lo["diff11"] - b11["diff"]
    lo["ch_dd"] = (lo["diff10"] - lo["diff11"]) - est
    lo["ch_low10"] = lo["low10"] - b10["gate_low"]
    summ = {}
    for k in ["ch_diff10", "ch_diff11", "ch_dd", "ch_low10"]:
        i = lo[k].abs().idxmax()
        summ[k] = {"iso": lo.loc[i, "iso"], "change": float(lo.loc[i, k]),
                   "min_after": float((lo[k] + (est if k == "ch_dd" else 0)).min()),
                   "max_after": float((lo[k] + (est if k == "ch_dd" else 0)).max())}
    summ["dd_range"] = [float((lo["diff10"] - lo["diff11"]).min()),
                        float((lo["diff10"] - lo["diff11"]).max())]
    out["loco"] = summ
    json.dump(out, open(RES / "vintage.json", "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in out if k != "configs"}, indent=1, default=float))


if __name__ == "__main__":
    main()
