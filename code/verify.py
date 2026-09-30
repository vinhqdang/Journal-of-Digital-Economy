"""Verify the frozen data snapshot and the headline results.

    python code/verify.py data      # SHA-256 checksums of the archived WDI snapshot
    python code/verify.py results   # assert that the stored results reproduce the reported numbers

Run `verify.py data` before rebuilding the panels.  `verify.py results` checks the numbers quoted
in the abstract and in the main tables against results/ with a tolerance that allows for the
last digit; it is run after the estimation scripts.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WDI = ROOT / "data" / "raw" / "wdi"

# numbers reported in the manuscript (rounded as printed)
EXPECTED = {
    ("A", "Baseline", "fedml"): -2.79, ("A", "Baseline", "fedml_se"): 1.30,
    ("A", "Baseline", "diff"): -0.67, ("B", "Baseline", "diff"): -0.25,
    ("C", "Baseline", "diff"): -5.17, ("A", "Baseline", "n_obs"): 4075,
}


def check_data():
    bad = 0
    for line in (WDI / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        h = hashlib.sha256((WDI / name).read_bytes()).hexdigest()
        ok = h == digest
        bad += not ok
        print(("OK  " if ok else "FAIL"), name)
    if bad:
        sys.exit(f"{bad} file(s) differ from the archived snapshot")
    print("all checksums match")


def check_results():
    s = pd.read_csv(ROOT / "results" / "empirical_summary.csv")
    bad = 0
    for (P, spec, col), v in EXPECTED.items():
        got = float(s[(s["panel"] == P) & (s["spec"] == spec)][col].iloc[0])
        tol = 0.5 if col == "n_obs" else 0.006
        ok = abs(got - v) <= tol
        bad += not ok
        print(("OK  " if ok else "FAIL"), P, spec, col, f"{got:.4f}", "expected", v)
    v = json.load(open(ROOT / "results" / "vintage.json"))
    print("vintage difference", round(v["difference"]["est"], 2), "bootstrap s.e.",
          round(v["difference"]["boot_se"], 2))
    if bad:
        sys.exit(f"{bad} number(s) differ")
    print("all checked numbers match")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "data"
    check_data() if what == "data" else check_results()
