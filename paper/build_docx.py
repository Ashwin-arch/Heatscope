#!/usr/bin/env python3
"""Render the editable DOCX edition of the HeatScope paper from paper.md.

Pandoc 3.x does not build LaTeX-style figures from Markdown caption syntax, so
figure blocks are rewritten here as an image plus an explicit caption paragraph
before conversion. The PDF edition is produced by build_pdf.py, which restores
proper float captions.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
MD = HERE / "paper.md"
OUT = HERE / "HeatScope_IEEE_Research_Paper.docx"

FIGURE = re.compile(
    r'^!\[(?P<alt>[^\]]*)\]\((?P<img>[^)]+)\)\{width=\\linewidth\}\n\n: (?P<cap>.+)$',
    flags=re.M,
)


def rewrite_figures(text: str) -> tuple[str, int]:
    counter = {"n": 0}

    def repl(m: re.Match) -> str:
        counter["n"] += 1
        return "![%s](%s){width=6.2in}\n\n**Fig. %d.** %s" % (
            m.group("alt"),
            m.group("img"),
            counter["n"],
            m.group("cap"),
        )

    return FIGURE.subn(repl, text)


def main() -> None:
    text = MD.read_text()
    text, n = rewrite_figures(text)
    staged = HERE / "_docx.md"
    staged.write_text(text)
    print(f"rewrote {n} figure captions for the DOCX edition")

    subprocess.run(
        [
            "pandoc",
            str(staged),
            "-f",
            "markdown-implicit_figures",
            "--resource-path",
            str(HERE),
            "-o",
            str(OUT),
        ],
        check=True,
    )
    staged.unlink()
    print(f"wrote {OUT.name} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()