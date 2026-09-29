"""Refresh the generated table blocks that are inlined in manuscript/main.tex."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tex = (ROOT / "manuscript" / "main.tex").read_text()
for f in (ROOT / "manuscript" / "tables").glob("*.tex"):
    pat = re.compile(r"(%% BEGIN tables/" + f.stem + r"\.tex[^\n]*\n).*?(\n%% END tables/" + f.stem + ")", re.S)
    tex = pat.sub(lambda m: m.group(1) + f.read_text().strip() + m.group(2), tex)
(ROOT / "manuscript" / "main.tex").write_text(tex)
