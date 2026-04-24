#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="sandbox_results/dataloader_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

BASE_CFG="config/sandbox/mlp_caloinn300_sandbox_baseline.yaml"
OPT_CFG="config/sandbox/mlp_caloinn300_sandbox_optimized.yaml"
LOG_FILE="$RESULTS_DIR/dataloader_benchmark.log"

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

uv run python -B scripts/sandbox/measure_dataloader.py \
  --baseline-config "$BASE_CFG" \
  --optimized-config "$OPT_CFG" \
  --steps 120 \
  --warmup 20 \
  --gpu-transfer \
  | tee "$LOG_FILE"

echo "Saved: $LOG_FILE"
