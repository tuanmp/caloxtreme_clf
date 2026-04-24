#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="sandbox_results/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

BASE_CFG="config/sandbox/mlp_caloinn300_sandbox_baseline.yaml"
OPT_CFG="config/sandbox/mlp_caloinn300_sandbox_optimized.yaml"

# Cluster file locking can fail on shared filesystems under contention.
export HDF5_USE_FILE_LOCKING=FALSE

COMMON_OVERRIDES=(
  --trainer.logger false
)

run_case() {
  local name="$1"
  local cfg="$2"
  local log_file="$RESULTS_DIR/${name}.log"

  echo "Running ${name} with ${cfg}" | tee -a "$RESULTS_DIR/summary.txt"
  local t0
  t0=$(date +%s)

  if ! uv run python main.py fit --config "$cfg" "${COMMON_OVERRIDES[@]}" >"$log_file" 2>&1; then
    echo "${name}: FAILED" | tee -a "$RESULTS_DIR/summary.txt"
    echo "---- tail ${name}.log ----" | tee -a "$RESULTS_DIR/summary.txt"
    tail -n 60 "$log_file" | tee -a "$RESULTS_DIR/summary.txt"
    return 1
  fi

  local t1
  t1=$(date +%s)
  local duration=$((t1 - t0))

  local speed
  speed=$(grep -oE '[0-9]+\.?[0-9]* it/s' "$log_file" | tail -1 | awk '{print $1}')
  if [[ -z "${speed}" ]]; then
    speed="N/A"
  fi

  echo "${name}: ${speed} it/s (${duration}s)" | tee -a "$RESULTS_DIR/summary.txt"
}

echo "Sandbox benchmark start: $(date)" | tee "$RESULTS_DIR/summary.txt"
echo "Host: $(hostname)" | tee -a "$RESULTS_DIR/summary.txt"

run_case "baseline" "$BASE_CFG"
run_case "optimized" "$OPT_CFG"

python3 - <<'PY' "$RESULTS_DIR/summary.txt"
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read()
vals = dict(re.findall(r"(baseline|optimized):\s+([0-9]+\.?[0-9]*)\s+it/s", text))
if "baseline" in vals and "optimized" in vals:
    b = float(vals["baseline"])
    o = float(vals["optimized"])
    if b > 0:
        print(f"speedup: {o / b:.3f}x ({(o / b - 1) * 100:.1f}%)")
PY

echo "Logs written to ${RESULTS_DIR}"
