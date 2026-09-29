"""Additional results appended to results/empirical.json after empirical.py.

* dose-response model in internet penetration (O'Sullivan penalty),
* two-way (country and year) clustered standard errors for the baseline functionals,
* Holm-adjusted p-values of the three pre-specified sup-t constancy tests (penalised fit),
* minimum detectable gradients for the schooling moderator.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from empirical import BASE, dose_response, fit_dose, run_spec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj.tolist()


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    js = json.load(open(RES / "empirical.json"))
    summ = pd.read_csv(RES / "empirical_summary.csv")
    ex = js["extra"]
    # screening robustness on the panel built from the unscreened adoption series
    uns = pd.read_csv(ROOT / "data" / "processed" / "panel_unscreened.csv")
    ex["unscreened_n"] = len(uns)
    for P, z in [("A", "hc_l1"), ("B", "lp_init")]:
        r, cv = run_spec({"panel": P, "name": "Without data screening", "data": uns,
                          "kw": {"z": z, "n_rep": 3}})
        idx = summ[(summ["panel"] == P) & (summ["spec"] == "Without data screening")].index
        for k, v in r.items():
            summ.loc[idx, k] = v
        js["curves"][f"{P}|Without data screening"] = cv
    summ.to_csv(RES / "empirical_summary.csv", index=False)
    ex["dose"] = dose_response(df)
    print("dose done", flush=True)

    tw = {}
    for P, z in [("A", "hc_l1"), ("B", "lp_init"), ("C", "ysince_l1")]:
        m = fit_dose(df, z=z)
        zz = m._Z
        q1, q2 = np.quantile(zz, [1 / 3, 2 / 3])
        l_diff = m._L_mean(zz > q2) - m._L_mean(zz <= q1)
        c = copy.copy(m)
        c._raw = m._raw
        c.rebasis(0, 0, penalty=False)
        tw[P] = {"diff_se_twoway": m.twoway_se(l_diff), "diff_se_country": m.group_contrast(
                     zz > q2, zz <= q1)[1],
                 "fedml_se_twoway": c.twoway_se(np.ones(1)), "fedml_se_country": c.average_effect()[1]}
        print("twoway", P, flush=True)
    ex["twoway"] = tw

    base = summ[summ["spec"] == "Baseline"].set_index("panel")
    # primary tests: sup-t constancy tests of the penalised fit (size-controlled in the plasmode)
    ps = [float(base.loc[P, "p_const"]) for P in ["A", "B", "C"]]
    pu = [float(base.loc[P, "p_const_u"]) for P in ["A", "B", "C"]]
    ex["holm"] = {"raw": ps, "adjusted": holm(ps), "raw_unpen": pu, "adjusted_unpen": holm(pu),
                  "panels": ["A", "B", "C"]}
    a = base.loc["A"]
    ex["mde_schooling"] = {"diff_mde": 2.8 * a["diff_se"], "blp_mde": 2.8 * a["blp_se"],
                           "diff_ci": [a["diff"] - 1.96 * a["diff_se"], a["diff"] + 1.96 * a["diff_se"]],
                           "blp_ci": [a["blp"] - 1.96 * a["blp_se"], a["blp"] + 1.96 * a["blp_se"]]}
    json.dump(js, open(RES / "empirical.json", "w"), indent=1, default=float)
    print(json.dumps({k: ex[k] for k in ["twoway", "holm", "mde_schooling"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
