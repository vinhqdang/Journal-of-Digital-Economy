"""Additional results appended to results/empirical.json after empirical.py.

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
from empirical import fit_dose, tercile_masks  # noqa: E402

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
    tw = {}
    for P, z in [("A", "hc_l1"), ("B", "lp_init"), ("C", "ysince_l1")]:
        d = df.dropna(subset=[z]).reset_index(drop=True)
        m = fit_dose(d, z=z)
        lo, _, hi = tercile_masks(m, d)
        l_diff = m._L_mean(hi) - m._L_mean(lo)
        c = copy.copy(m)
        c._raw = m._raw
        c.rebasis(0, 0, penalty=False)
        tw[P] = {"diff_se_twoway": m.twoway_se(l_diff), "diff_se_country": m.group_contrast(hi, lo)[1],
                 "fedml_se_twoway": c.twoway_se(np.ones(1)), "fedml_se_country": c.average_effect()[1]}
        print("twoway", P, flush=True)
    ex["twoway"] = tw

    base = summ[summ["spec"] == "Baseline"].set_index("panel")
    # primary tests: sup-t constancy tests of the penalised fit (size-controlled in the plasmode)
    ps = [float(base.loc[P, "p_const"]) for P in ["A", "B", "C"]]
    pu = [float(base.loc[P, "p_const_u"]) for P in ["A", "B", "C"]]
    pl = [float(base.loc[P, "p_lin"]) for P in ["A", "B", "C"]]
    ex["holm"] = {"raw": ps, "adjusted": holm(ps), "raw_unpen": pu, "adjusted_unpen": holm(pu),
                  "raw_lin": pl, "adjusted_lin": holm(pl), "panels": ["A", "B", "C"]}
    ex["mde"] = {}
    for P in ["A", "B", "C"]:
        a = base.loc[P]
        ex["mde"][P] = {"diff_mde": 2.8 * a["diff_se"], "blp_mde": 2.8 * a["blp_se"],
                        "diff_ci": [a["diff"] - 1.96 * a["diff_se"], a["diff"] + 1.96 * a["diff_se"]],
                        "blp_ci": [a["blp"] - 1.96 * a["blp_se"], a["blp"] + 1.96 * a["blp_se"]]}
    ex["mde_schooling"] = ex["mde"]["A"]
    json.dump(js, open(RES / "empirical.json", "w"), indent=1, default=float)
    print(json.dumps({k: ex[k] for k in ["twoway", "holm", "mde"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
