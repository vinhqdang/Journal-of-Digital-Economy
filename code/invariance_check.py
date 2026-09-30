"""Numerical check of Proposition 1(ii) on the baseline sample (appendix, 'Invariance check').

Country-specific constants are added to the cross-fitted residuals of the baseline fits and the
final stage is re-solved with the penalty held at its selected value.  Output:
results/invariance.json.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from empirical import fit_dose  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def contrast(m):
    z = m._Z
    q1, q2 = np.quantile(z, [1 / 3, 2 / 3])
    return m.group_contrast(z > q2, z <= q1)


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "panel.csv")
    out = {}
    rng = np.random.default_rng(3)
    for P, z in [("A", "hc_l1"), ("B", "lp_init")]:
        m = fit_dose(df, z=z)
        lam = m.lambda_
        m.refit_final(lambda_fixed=lam)
        base_d, base_se = contrast(m)
        c = copy.copy(m)
        c._raw = m._raw
        c.rebasis(0, 0, penalty=False)
        base_c = c.average_effect()[0]
        codes = pd.factorize(m.unit_)[0]
        sd_rd = float(np.std(m._raw[0][1]))
        res = {"lambda": lam, "contrast": base_d, "contrast_se": base_se, "fedml": base_c,
               "sd_treatment_residual": sd_rd}
        for what, scale in [("outcome", 1.0), ("treatment", 0.05)]:
            shift = rng.normal(0, scale, codes.max() + 1)[codes]
            raws = [(ry + shift, rd) if what == "outcome" else (ry, rd + shift)
                    for ry, rd in m._raw]
            mm = copy.copy(m)
            mm._raw = raws
            mm.rebasis(4, 3, penalty=True)
            d, _ = contrast(mm)
            cc = copy.copy(m)
            cc._raw = raws
            cc.rebasis(0, 0, penalty=False)
            res[f"{what}_contrast"] = d
            res[f"{what}_fedml"] = cc.average_effect()[0]
        out[P] = res
    json.dump(out, open(ROOT / "results" / "invariance.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
