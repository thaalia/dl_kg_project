#!/usr/bin/env bash
# Build PDFs for all submissions and the poster (from repo root:  bash submissions/compile.sh)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEX1="$ROOT/submissions/deep_learning/milestone1/milestone1_dl.tex"
TEX2="$ROOT/submissions/knowledge_graphs/d2/kg_report_d2.tex"
TEX3="$ROOT/submissions/knowledge_graphs/d3/kg_report_d3.tex"
for t in "$TEX1" "$TEX2" "$TEX3"; do
  echo "=== $t ==="
  ( cd "$(dirname "$t")" && pdflatex -interaction=nonstopmode "$(basename "$t")" && bibtex "$(basename "$t" .tex)" && pdflatex -interaction=nonstopmode "$(basename "$t")" && pdflatex -interaction=nonstopmode "$(basename "$t")" )
done

echo "=== $ROOT/poster/poster.tex ==="
( cd "$ROOT/poster" && pdflatex -interaction=nonstopmode poster.tex && pdflatex -interaction=nonstopmode poster.tex )

echo "OK: check milestone1_dl.pdf, kg_report_d2.pdf, kg_report_d3.pdf, and poster/poster.pdf"
