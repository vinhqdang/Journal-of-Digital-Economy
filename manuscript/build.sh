#!/bin/sh
# Build the manuscript, the supplementary material (full and anonymised) and the title page.
cd "$(dirname "$0")"
for d in main supplement main_blind supplement_blind; do pdflatex -interaction=nonstopmode $d >/dev/null; done
for d in main supplement main_blind supplement_blind; do bibtex $d >/dev/null; done
for i in 1 2; do
  for d in main supplement main_blind supplement_blind; do pdflatex -interaction=nonstopmode $d >/dev/null; done
done
pdflatex -interaction=nonstopmode title_page >/dev/null
grep -l "^!" *.log || echo "no LaTeX errors"
