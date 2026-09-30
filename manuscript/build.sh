#!/bin/sh
# Build the manuscript, the supplementary material (full and anonymised) and the title page.
cd "$(dirname "$0")"
for d in main supplement main_blind supplement_blind; do pdflatex -interaction=nonstopmode $d >/dev/null; done
for d in main supplement main_blind supplement_blind; do bibtex $d >/dev/null; done
for i in 1 2; do
  for d in main supplement main_blind supplement_blind; do pdflatex -interaction=nonstopmode $d >/dev/null; done
done
pdflatex -interaction=nonstopmode title_page >/dev/null
pdflatex -interaction=nonstopmode cover_letter >/dev/null
grep -l "^!" *.log || echo "no LaTeX errors"
# Copies with descriptive names for submission (the anonymised files carry no author name)
mkdir -p ../submission
cp main_blind.pdf ../submission/Digital_Dividend_Manuscript_anonymised.pdf
cp supplement_blind.pdf ../submission/Digital_Dividend_Supplementary_Material_anonymised.pdf
cp title_page.pdf ../submission/Digital_Dividend_Title_Page.pdf
cp cover_letter.pdf ../submission/Digital_Dividend_Cover_Letter.pdf
cp main.pdf ../submission/Digital_Dividend_Manuscript_with_author_details.pdf
cp supplement.pdf ../submission/Digital_Dividend_Supplementary_Material_with_author_details.pdf
