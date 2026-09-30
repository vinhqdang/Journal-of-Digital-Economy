"""Build an anonymised copy of the replication package for double-blind review.

The archive contains the code, the data snapshot, the results and the anonymised manuscript
sources; author names, affiliations and repository addresses are removed from the README and
the package metadata, and the non-anonymised manuscript files are left out.

Output: submission/replication_anonymous.zip
"""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"
OUT.mkdir(exist_ok=True)

INCLUDE_DIRS = ["code", "data", "results", "manuscript/tables", "manuscript/figures"]
INCLUDE_FILES = ["requirements.txt", "requirements-lock.txt", "LICENSE",
                 "manuscript/preamble.tex", "manuscript/main.tex", "manuscript/supplement.tex",
                 "manuscript/main_blind.tex",
                 "manuscript/supplement_blind.tex", "manuscript/refs.bib",
                 "manuscript/build.sh"]
EXCLUDE = {"code/make_anonymous_package.py", "code/make_springer.py"}


def scrub(text):
    """Remove identifying strings (names, affiliation, e-mail, ORCID, repository URL)."""
    pats = [r"Quang-Vinh Dang", r"Dang, Quang-Vinh", r"British University Vietnam", r"Hung Yen",
            r"vinh\.dq4@buv\.edu\.vn", r"0000-0002-3877-8024", r"vinhqdang", r"dqvinh87@gmail\.com"]
    for p in pats:
        text = re.sub(p, "[anonymised]", text)
    return text


def readme():
    t = (ROOT / "README.md").read_text()
    t = re.sub(r"> Quang-Vinh Dang.*\n", "> (author details withheld for double-blind review)\n", t)
    return scrub(t)


def pyproject():
    t = (ROOT / "pyproject.toml").read_text()
    t = re.sub(r"authors\s*=\s*\[[^\]]*\]\n?", "", t)
    return scrub(t)


def main():
    path = OUT / "replication_anonymous.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("replication/README.md", readme())
        z.writestr("replication/pyproject.toml", pyproject())
        for d in INCLUDE_DIRS:
            for f in sorted((ROOT / d).rglob("*")):
                rel = f.relative_to(ROOT).as_posix()
                if f.is_file() and rel not in EXCLUDE and "__pycache__" not in rel:
                    if f.suffix in {".py", ".tex", ".md", ".txt", ".toml"}:
                        z.writestr("replication/" + rel, scrub(f.read_text()))
                    else:
                        z.write(f, "replication/" + rel)
        for rel in INCLUDE_FILES:
            f = ROOT / rel
            if f.exists():
                z.writestr("replication/" + rel, scrub(f.read_text()))
    # final check: no identifying string left in any text member
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith((".py", ".tex", ".md", ".txt", ".toml", ".bib", ".sh", ".csv", ".json")):
                t = z.read(n).decode("utf-8", "ignore")
                for s in ["Quang-Vinh", "vinhqdang", "buv.edu", "0000-0002-3877-8024"]:
                    assert s not in t, (n, s)
    print("wrote", path, round(path.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
