#!/usr/bin/env bash
set -euo pipefail

python benchmarks/scripts/compute_anchor_density.py
python benchmarks/scripts/validate_pairs_orthogroups.py
python benchmarks/scripts/analyze_cosine_vs_identity.py --n-pairs 1500
python benchmarks/scripts/run_threshold_sweep.py
python benchmarks/scripts/run_threshold_sweep.py \
  --genes-parquet elsa_index_borg/ingest/genes.parquet \
  --species all \
  --no-operon \
  --output benchmarks/evaluation/threshold_sweep_borg_summary.csv
python benchmarks/scripts/render_genome_browser_stub.py
python benchmarks/scripts/build_manuscript_figures_v3.py

MANUSCRIPT=benchmarks/evaluation/manuscript/ELSA_manuscript_v3.md
HTML=benchmarks/evaluation/manuscript/ELSA_manuscript_v3.html
PDF=benchmarks/evaluation/manuscript/ELSA_manuscript_v3.pdf

PANDOC_CMD=()
PDF_ENGINE=""

if command -v pandoc >/dev/null 2>&1 && pandoc --version >/dev/null 2>&1; then
  PANDOC_CMD=(pandoc)
  if command -v weasyprint >/dev/null 2>&1 && weasyprint --version >/dev/null 2>&1; then
    PDF_ENGINE="weasyprint"
  elif command -v wkhtmltopdf >/dev/null 2>&1 && wkhtmltopdf --version >/dev/null 2>&1; then
    PDF_ENGINE="wkhtmltopdf"
  fi
elif command -v conda >/dev/null 2>&1 && conda run -n elsa pandoc --version >/dev/null 2>&1; then
  PANDOC_CMD=(conda run -n elsa pandoc)
  if conda run -n elsa weasyprint --version >/dev/null 2>&1; then
    PDF_ENGINE="weasyprint"
  elif conda run -n elsa wkhtmltopdf --version >/dev/null 2>&1; then
    PDF_ENGINE="wkhtmltopdf"
  fi
elif command -v pyenv >/dev/null 2>&1; then
  PYENV_ROOT="$(pyenv root 2>/dev/null || true)"
  if [[ -n "$PYENV_ROOT" ]]; then
    ENV_BIN="$PYENV_ROOT/versions/miniconda3-latest/envs/elsa/bin"
    if [[ -x "$ENV_BIN/pandoc" ]]; then
      export PATH="$ENV_BIN:$PATH"
      PANDOC_CMD=(pandoc)
      if [[ -x "$ENV_BIN/weasyprint" ]]; then
        PDF_ENGINE="weasyprint"
      elif [[ -x "$ENV_BIN/wkhtmltopdf" ]]; then
        PDF_ENGINE="wkhtmltopdf"
      fi
    fi
  fi
fi

if ((${#PANDOC_CMD[@]})); then
  "${PANDOC_CMD[@]}" "$MANUSCRIPT" -o "$HTML" --resource-path="benchmarks/evaluation/manuscript"
  if [[ -n "$PDF_ENGINE" ]]; then
    "${PANDOC_CMD[@]}" "$MANUSCRIPT" -o "$PDF" --pdf-engine="$PDF_ENGINE" --resource-path="benchmarks/evaluation/manuscript"
  else
    echo "pandoc available, but no PDF engine (weasyprint/wkhtmltopdf). Skipping PDF."
  fi
else
  echo "pandoc not found (or conda env missing); skipping HTML/PDF render."
fi
