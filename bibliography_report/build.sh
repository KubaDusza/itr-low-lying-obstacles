#!/bin/zsh
# REPORT.md -> REPORT.tex (IEEEtran) -> REPORT.pdf, bibliography from refs.bib
set -e
cd "$(dirname "$0")"
pandoc REPORT.md -f markdown -t latex --template=template.tex --lua-filter=ieee.lua -o REPORT.tex
run() { pdflatex -interaction=nonstopmode -halt-on-error REPORT.tex >/dev/null || { tail -30 REPORT.log; exit 1; } }
run; bibtex REPORT >/dev/null || { cat REPORT.blg; exit 1; }; run; run
grep -i "warning.*undefined" REPORT.log || true
rm -f REPORT.aux REPORT.log REPORT.out REPORT.bbl REPORT.blg
echo "pages: $(pdfinfo REPORT.pdf | awk '/^Pages/{print $2}')"
