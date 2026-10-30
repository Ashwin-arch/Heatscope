#!/usr/bin/env python3
"""Assemble an IEEE-formatted LaTeX document from paper.md and compile it."""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
MD = HERE / "paper.md"
KEYWORDS = (
    "urban heat island, surface temperature, remote sensing, explainable AI, "
    "spatial cross-validation, equity-aware optimisation, urban heat vulnerability"
)

PREAMBLE = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{calc}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage[hyphens]{url}
\usepackage{textcomp}
\providecommand{\pandocbounded}[1]{#1}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\graphicspath{{figures/}}
\begin{document}
\title{HeatScope: An Explainable, Spatially Validated and Equity-Aware Framework for Mapping Urban Heat Exposure and Prioritising Mitigation Interventions}
\author{\IEEEauthorblockN{Ashwin Kumar G Rao}
\IEEEauthorblockA{Departments of Computer Science and Data Science\\
Manipal Institute of Technology\\
Bangalore, India}}
\maketitle
"""


def extract_abstract(text: str) -> str:
    m = re.search(
        r"# Abstract\n(.*?)\n\*\*Index Terms\*\*\u2014(.*?)\.\n", text, flags=re.S
    )
    if not m:
        sys.exit("could not locate the Abstract section")
    return m.group(1).strip()


LONG_TABLE = re.compile(
    r"(?P<lead>\\textbf\{Table [IVX]+\.[^\n]*\}\n\n)?"
    r"\{\\def\\LTcaptype\{none\} % do not increment counter\n"
    r"\\begin\{longtable\}\[\]\{(?P<spec>.*?)\}\n"
    r"(?P<content>.*?)"
    r"\\end\{longtable\}\n\}",
    flags=re.S,
)


def latex_tables(body: str) -> tuple[str, int]:
    """Convert pandoc longtable output into IEEEtran table floats."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        count += 1
        spec = re.sub(r"\s+", " ", m.group("spec")).strip()
        content = m.group("content")
        content = re.sub(r"\\endhead\n?", "", content)
        content = re.sub(r"\\endfirsthead\n?", "", content)
        content = re.sub(r"\\bottomrule\\noalign\{\}\n\\endlastfoot\n?", "", content)
        content = re.sub(r"\\bottomrule\\noalign\{\}\n\\endfoot\n?", "", content)
        content = content.strip()
        caption = ""
        if m.group("lead"):
            caption = m.group("lead").strip()[len("\\textbf{") : -len("}")]
            caption = re.sub(r"^Table [IVX]+\.\s*", "", caption)
        env = "table*" if ("p{" in spec or count in WIDE_TABLES) else "table"
        return (
            "\\begin{%s}[!t]\n"
            "\\caption{%s}\n"
            "\\label{tab:%d}\n"
            "\\centering\n"
            "\\footnotesize\n"
            "\\begin{tabular}{%s}\n"
            "%s\n"
            "\\end{tabular}\n"
            "\\end{%s}\n" % (env, caption, count, spec, content, env)
        )

    return LONG_TABLE.subn(repl, body)


LATEX_FIXES = {
    "\u2212": r"\textminus{}",  # minus sign
    "\u00d7": r"\texttimes{}",  # multiplication sign
    "\u00b7": r"\textperiodcentered{}",  # middle dot
    "\u2265": r"\ensuremath{\geq}",
    "\u2264": r"\ensuremath{\leq}",
    "\u2248": r"\ensuremath{\approx}",
    "\u2192": r"\ensuremath{\rightarrow}",
    "\u00b0": r"\textdegree{}",
    "\u00b1": r"\textpm{}",
}


def sanitize_latex(tex: str) -> str:
    """Replace Unicode symbols that pdflatex cannot typeset directly."""
    for char, repl in LATEX_FIXES.items():
        tex = tex.replace(char, repl)
    return tex


HEADING_NUM = re.compile(r"(\\(?:sub)?section)\{[IVX]+\.\s+")
SUBSECTION_NUM = re.compile(r"(\\subsection)\{[A-Z]\.\s+")
BIB_ENTRY = re.compile(r"^\{\[\}(\d+)\{\]\}\s", flags=re.M)

# Tables 2 and 6 hold wide numeric blocks that do not fit in one IEEE column.
WIDE_TABLES = {2, 6}

# IEEEtran conference letter paper: \columnwidth = 252 pt = 3.5 in.
SINGLE_COLUMN_IN = 3.5


def figure_width_in(rel_path: str) -> float:
    """Printed width of a figure PNG in inches, from its pixel size at 300 dpi."""
    from PIL import Image

    path = HERE / rel_path
    if not path.exists():
        sys.exit(f"missing figure: {path}")
    with Image.open(path) as im:
        return im.size[0] / 300.0


def ieee_structure(body: str) -> str:
    """Normalise pandoc output for IEEEtran: headings, abstract, bibliography."""
    # The abstract and index terms are emitted by the template instead.
    body = re.sub(
        r"\\section\{Abstract\}\\label\{abstract\}\n.*?(?=\\section\{)", "", body, flags=re.S
    )
    body = re.sub(
        r"\\textbf\{Index Terms\}---.*?\n\n", "", body, count=1, flags=re.S
    )
    # IEEEtran numbers sections itself; drop the hand-written numbers.
    body = SUBSECTION_NUM.sub(lambda m: "%s{" % m.group(1), body)
    body = HEADING_NUM.sub(lambda m: "%s{" % m.group(1), body)

    # Manual reference list -> IEEE bibliography.
    m = re.search(r"\\section\{References\}\\label\{references\}\n\n", body)
    if not m:
        sys.exit("could not locate the reference list")
    head, refs = body[: m.start()], body[m.end() :]
    refs = refs.strip()
    refs = BIB_ENTRY.sub(lambda mm: "\n\\bibitem{ref%s} " % mm.group(1), refs)
    if "\\bibitem" not in refs:
        sys.exit("reference entries were not recognised")
    n = refs.count("\\bibitem")
    head += (
        "\\begin{thebibliography}{99}\n" + refs.strip() + "\n\\end{thebibliography}\n"
    )
    print(f"converted {n} references into an IEEE bibliography")
    return head


DESCRIPTION_FIG = re.compile(
    r"\\begin\{description\}\n(?P<tight>\\tightlist\n)?"
    r"\\item\[\\pandocbounded\{\\includegraphics\[[^\]]*\]\{(?P<path>[^}]+)\}\}\]\s*"
    r"(?P<cap>.*?)"
    r"\\end\{description\}",
    flags=re.S,
)


def latex_figures(body: str) -> tuple[str, int]:
    """Convert pandoc description lists of images into IEEEtran figure floats.

    A figure wider than SINGLE_COLUMN_IN cannot be read at column width (an
    IEEE column is 3.5 in), so anything wider becomes a two-column float.
    """
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        count += 1
        cap = m.group("cap").strip()
        if "\\begin{" in cap:
            sys.exit("figure caption contains a nested environment:\n" + cap)
        env = (
            "figure*"
            if figure_width_in(m.group("path")) > SINGLE_COLUMN_IN
            else "figure"
        )
        return (
            "\\begin{%s}[!t]\n"
            "\\centering\n"
            "\\includegraphics[width=\\linewidth]{%s}\n"
            "\\caption{%s}\n"
            "\\label{fig:%d}\n"
            "\\end{%s}\n" % (env, m.group("path"), cap, count, env)
        )

    return DESCRIPTION_FIG.subn(repl, body)


def main() -> None:
    text = MD.read_text()
    abstract_md = extract_abstract(text)

    subprocess.run(
        [
            "pandoc",
            str(MD),
            "-f",
            "markdown-implicit_figures",
            "-t",
            "latex",
            "--wrap=preserve",
            "-o",
            str(HERE / "_body.tex"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    body = (HERE / "_body.tex").read_text()
    body = sanitize_latex(body)
    body, n_tables = latex_tables(body)
    body, n_figures = latex_figures(body)
    body = ieee_structure(body)
    (HERE / "_body.tex").write_text(body)
    print(f"converted {n_tables} longtables into table floats")
    print(f"converted {n_figures} image lists into figure floats")
    if "description}" in body or "pandocbounded" in body:
        sys.exit("unconverted figure markup remains")
    if re.search(r"^\\textbf\{Table ", body, flags=re.M) or "\\begin{longtable}" in body:
        sys.exit("unconverted table markup remains")

    abstract_tex = subprocess.run(
        ["pandoc", "-f", "markdown", "-t", "latex", "--wrap=preserve"],
        input=abstract_md,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    doc = (
        PREAMBLE
        + "\\begin{abstract}\n"
        + abstract_tex
        + "\n\\end{abstract}\n\n"
        + "\\begin{IEEEkeywords}\n"
        + KEYWORDS
        + "\n\\end{IEEEkeywords}\n\n"
        + body
        + "\n\\end{document}\n"
    )
    out = HERE / "HeatScope_IEEE_Research_Paper.tex"
    out.write_text(doc)
    print(f"assembled {out.name} ({len(doc):,} bytes)")

    for engine in ("pdflatex",):
        for _ in range(2):
            r = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", out.stem + ".tex"],
                cwd=HERE,
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                print(r.stdout[-4000:])
                print(r.stderr[-2000:])
                sys.exit(f"{engine} failed")
    pdf = HERE / "HeatScope_IEEE_Research_Paper.pdf"
    shutil.move(str(pdf), str(HERE / "HeatScope_IEEE_Research_Paper.pdf"))
    pages = subprocess.run(
        ["pdfinfo", str(HERE / "HeatScope_IEEE_Research_Paper.pdf")],
        capture_output=True,
        text=True,
    ).stdout
    print(pages.strip()[:400])


if __name__ == "__main__":
    main()
