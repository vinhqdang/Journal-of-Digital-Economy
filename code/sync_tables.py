"""Refresh the generated table blocks inlined in manuscript/main.tex and supplement.tex."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for doc in ["main.tex", "supplement.tex"]:
    path = ROOT / "manuscript" / doc
    tex = path.read_text()
    for f in (ROOT / "manuscript" / "tables").glob("*.tex"):
        pat = re.compile(r"(%% BEGIN tables/" + f.stem + r"\.tex[^\n]*)\n.*?\n?(%% END tables/"
                         + f.stem + ")", re.S)
        tex = pat.sub(lambda m: m.group(1) + "\n" + f.read_text().strip() + "\n" + m.group(2), tex)
    path.write_text(tex)
