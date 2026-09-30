"""Build the Springer Nature (sn-jnl) version of the manuscript from main.tex.

The body text, tables and appendix are taken verbatim from manuscript/main.tex;
only the preamble, front matter, declarations and bibliography style change.
Output: manuscript/springer/main_sn.tex (compile inside manuscript/springer).
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "manuscript" / "main.tex"
SUP = ROOT / "manuscript" / "supplement.tex"
OUT = ROOT / "manuscript" / "springer" / "main_sn.tex"

PREAMBLE = r"""\documentclass[pdflatex,sn-basic]{sn-jnl}% Springer Nature author-year reference style

\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage[title]{appendix}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
\usepackage{bm}
\usepackage{adjustbox}
\graphicspath{{../figures/}{../}}

\theoremstyle{thmstyleone}
\newtheorem{assumption}{Assumption}
\newtheorem{proposition}{Proposition}
\newtheorem{lemma}{Lemma}
\theoremstyle{thmstyletwo}
\newtheorem{remark}{Remark}
\newcommand{\E}{\mathbb{E}}
\newcommand{\M}{\mathcal{M}}
\DeclareMathOperator*{\argmin}{arg\,min}
\input{../tables/numbers.tex}
\newif\ifblind
\newcommand{\externaldocument}[1]{}
% Keep the sn-jnl table fonts but drop its automatic threeparttable wrapper,
% so that wide tables can be scaled to the text width.
\renewenvironment{table}[1][]{\begin{tableorg}[#1]\centering\tablebodyfont}{\end{tableorg}}

\raggedbottom

\begin{document}

\title[Searching for the digital dividend]{TITLE}

\author*[1]{\fnm{Quang-Vinh} \sur{Dang}}\email{vinh.dq4@buv.edu.vn}

\affil*[1]{\orgname{British University Vietnam}, \orgaddress{\city{Hung Yen}, \country{Vietnam}}}

\abstract{ABSTRACT}

\keywords{KEYWORDS}

\pacs[JEL Classification]{JELCODES}

\maketitle
"""


def grab(pattern, text):
    m = re.search(pattern, text, re.S)
    if not m:
        raise ValueError(pattern)
    return m.group(1).strip()


def main():
    tex = SRC.read_text()
    title = grab(r"\\title\{(.*?)\}\n", tex)
    abstract = grab(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex)
    kw = grab(r"\\begin\{keyword\}(.*?)\\JEL", tex)
    jel = grab(r"\\JEL(.*?)\\end\{keyword\}", tex)
    kw = ", ".join(k.strip() for k in kw.split(r"\sep"))
    jel = ", ".join(k.strip() for k in jel.split(r"\sep"))

    body = tex.split(r"\end{frontmatter}", 1)[1]
    main_part = body.split(r"\bibliographystyle{elsarticle-harv}", 1)[0]
    sup = SUP.read_text()
    appendix = sup.split(r"\appendix", 1)[1].split(r"\bibliographystyle{elsarticle-harv}", 1)[0]
    main_part = main_part.replace("The supplementary material contains", "The appendices contain")
    # the KeAi version wraps the bibliography in a single-spacing group
    main_part = main_part.rstrip().removesuffix("{\\singlespacing").rstrip()
    appendix = appendix.rstrip().removesuffix("{\\singlespacing").rstrip().removesuffix("\\clearpage")

    # Declarations in the Springer Nature format.
    main_part, decl = main_part.split(r"\section*{Data and code availability}", 1)
    decl = r"\section*{Data and code availability}" + decl
    items = re.findall(r"\\section\*\{(.*?)\}\n(.*?)(?=\\section\*|\Z)", decl, re.S)
    lines = ["\\backmatter\n", "\\section*{Declarations}\n"]
    for head, text in items:
        head = {"Declaration of competing interest": "Competing interests",
                "CRediT authorship contribution statement": "Author contribution",
                "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process":
                "Use of generative AI"}.get(head, head)
        lines.append("\\bmhead{%s}\n%s\n" % (head, text.strip()))
    decl = "\n".join(lines)

    appendix = appendix.replace(r"\section*{Online appendix}", "", 1)
    head = (PREAMBLE.replace("TITLE", title).replace("ABSTRACT", abstract)
            .replace("KEYWORDS", kw).replace("JELCODES", jel))
    # The sn-jnl text block is narrower; shrink wide tables to the line width.
    fit = lambda t: re.sub(r"(\\begin\{threeparttable\}.*?\\end\{threeparttable\})",
                           r"\\adjustbox{max width=\\linewidth}{%\n\1}", t, flags=re.S)
    main_part, appendix = fit(main_part), fit(appendix)
    # the journal-specific references of the KeAi version
    main_part = main_part.replace("Recent contributions to this journal",
                                  r"Recent contributions to the \textit{Journal of Digital Economy}")
    main_part = main_part.replace("increasingly used in this journal",
                                  r"increasingly used in the \textit{Journal of Digital Economy}")
    main_part = main_part.replace(r"(\ref{app:", r"(Appendix~\ref{app:")
    out = (head + main_part.rstrip() + "\n\n" + decl + "\n"
           + "\\begin{appendices}\n" + appendix.strip() + "\n\\end{appendices}\n\n"
           + "\\bibliography{../refs}\n\n\\end{document}\n")
    OUT.write_text(out)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
