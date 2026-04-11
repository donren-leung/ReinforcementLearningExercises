#!/usr/bin/env bash
set -euo pipefail

PY=python3
SCRIPT="dp.main"
OUT_BASE="dp/results"

modes=("eval" "iter" "value")
# modes=("value")
envs=("gamblers-0.25" "gamblers-0.45" "gamblers-0.50" "gamblers-0.55")
# envs=("escape" "jumping" "jacks" "jacks-small" "modjacks")

mkdir -p "$OUT_BASE"

for env in "${envs[@]}"; do
  env_out="$OUT_BASE/$env"
  mkdir -p "$env_out"
  for mode in "${modes[@]}"; do
      suffix="$mode"

      results_folder="$env_out/$suffix"
      mkdir -p "$results_folder"

      echo "Running: $PY $SCRIPT $results_folder --mode $mode -e $env"
      $PY -m "$SCRIPT" "$results_folder" --mode "$mode" -e "$env"
    done
done