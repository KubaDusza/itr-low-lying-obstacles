#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export PATH="/Library/TeX/texbin:$PATH"
export TEXINPUTS=".:vendor//:${TEXINPUTS:-}"
export BSTINPUTS=".:${BSTINPUTS:-}"
rm -f REPORT.bbl REPORT.aux
for step in 1 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex > build.log 2>&1 || { tail -35 build.log; exit 1; }
  if [[ "$step" == 1 ]]; then
    bibtex REPORT > bibliography_build.log 2>&1 || { cat bibliography_build.log; exit 1; }
  fi
done
pdfinfo REPORT.pdf | head -18
rg 'Overfull|undefined|LaTeX Warning' REPORT.log || true
