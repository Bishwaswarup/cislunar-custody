#!/usr/bin/env bash
# Build paper/main.pdf, or package the paper for Overleaf.
#   bash scripts/build_paper.sh            # compile locally if a TeX engine is installed, else make the zip
#   bash scripts/build_paper.sh --zip      # only make paper/overleaf.zip (upload: Overleaf > New project > Upload)
# Local engines, in order: tectonic (brew install tectonic; fetches packages on first use, needs
# internet once), latexmk, pdflatex + bibtex.
set -euo pipefail
cd "$(dirname "$0")/../paper"

make_zip() {
  rm -f overleaf.zip
  tmp=$(mktemp -d)
  mkdir -p "$tmp/figures/jas"
  cp main.tex references.bib sn-jnl.cls sn-mathphys-num.bst "$tmp/"
  cp ../figures/jas/Fig*.pdf "$tmp/figures/jas/"
  (cd "$tmp" && zip -qr overleaf.zip main.tex references.bib sn-jnl.cls sn-mathphys-num.bst figures)
  mv "$tmp/overleaf.zip" .
  rm -rf "$tmp"
  echo "paper/overleaf.zip ($(du -h overleaf.zip | cut -f1)): upload it on overleaf.com (New project > Upload project)."
  echo "Compiler: pdfLaTeX (Menu > Settings). Main document: main.tex."
}

if [[ "${1:-}" == "--zip" ]]; then
  make_zip
  exit 0
fi

if command -v tectonic >/dev/null; then
  tectonic --keep-logs main.tex
elif command -v latexmk >/dev/null; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
elif command -v pdflatex >/dev/null && command -v bibtex >/dev/null; then
  pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
  bibtex main >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null
else
  echo "No TeX engine found (try: brew install tectonic). Making the Overleaf zip instead."
  make_zip
  exit 0
fi
grep -E "undefined|Rerun to get" main.log >/dev/null 2>&1 && echo "warning: undefined references (see paper/main.log)" || true
echo "paper/main.pdf built"
