#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODE="${1:-formal}"
if [[ "$MODE" != "check" && "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
    printf 'Usage: bash scripts/experiments/run_osa_fusion_capacity_controls.sh check|smoke|formal\n' >&2
    exit 2
fi
mkdir -p logs
if command -v flock >/dev/null 2>&1; then
    exec 9>"logs/osa_stage50_${MODE}.lock"
    if ! flock -n 9; then
        printf 'Stage50 %s is already running; no second process started.\n' "$MODE" >&2
        exit 1
    fi
fi

PYTHON_BIN="${OSA_PYTHON:-$(command -v python)}"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
DEVICE="${OSA_DEVICE:-cpu}"
RUNNER="model/osa_adapter/osa_fusion_capacity_experiments.py"

printf '[START] Stage50 mode=%s Python=%s device=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$MODE" "$PYTHON_BIN" "$DEVICE" "${CUDA_VISIBLE_DEVICES:-not set}"
"$PYTHON_BIN" -c 'import sys, numpy, pandas, scipy, torch; print("Interpreter:", sys.executable); print("Dependencies: PASS"); print("Torch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_osa_fusion_capacity*.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_osa_fusion_external_methods.py' -v

if [[ "$MODE" == "check" ]]; then
    "$PYTHON_BIN" "$RUNNER" --check-env --device cpu
    exit 0
fi

EXTRA=()
if [[ "$MODE" == "smoke" ]]; then
    EXTRA+=(--smoke)
fi
"$PYTHON_BIN" "$RUNNER" --device "$DEVICE" --threads 1 --resume "${EXTRA[@]}"
"$PYTHON_BIN" "$RUNNER" --external-only --resume "${EXTRA[@]}"
printf '[COMPLETE] Stage50 mode=%s\n' "$MODE"
