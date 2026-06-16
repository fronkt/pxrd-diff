#!/usr/bin/env bash
# Reproducible build: paper.md (canonical source) -> paper.tex -> paper.pdf
# Requires pandoc (>=3) and tectonic (>=0.15, XeTeX). Run from anywhere.
#
#   pandoc  : apt-get install -y pandoc   (or https://pandoc.org/installing.html)
#   tectonic: https://tectonic-typesetting.github.io/  (single static binary)
#
# References in paper.md are pandoc footnotes, so no bibtex/citeproc step is needed.
set -euo pipefail
cd "$(dirname "$0")"   # paper/

pandoc paper.md -H _pandoc_header.tex -s --shift-heading-level-by=-1 \
  -V geometry:margin=1in -V documentclass=article -V fontsize=11pt \
  -o paper.tex

tectonic paper.tex
echo "Built $(pwd)/paper.pdf"
