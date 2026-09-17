#!/usr/bin/env bash
# Wait for enough free GPU memory, then atomically launch the registered
# REM-Net phase-interaction pilot. This script never kills another process.

set -euo pipefail

PROJECT_DIR="/home/qhh2237/Projects/BeatAware-Radar2ECGNet"
PYTHON_BIN="/home/qhh2237/miniconda3/envs/cyberbrain/bin/python"
RUNNER="scripts/run_remnet_phase_interaction_gate.py"
QUEUE_LOG="experiments_mmecg/q1_remnet_phase_interaction_gate_queue.log"
RUN_LOG="experiments_mmecg/q1_remnet_phase_interaction_gate_run.log"
LOCK_FILE="/tmp/beataware_remnet_phase_gate.lock"

MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GPU_INDEX="${GPU_INDEX:-0}"

cd "$PROJECT_DIR"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[queue] another REM-Net phase-gate queue already holds $LOCK_FILE" >&2
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[queue] nvidia-smi is unavailable" >&2
    exit 3
fi
if [[ ! -f "$RUNNER" ]]; then
    echo "[queue] runner is missing: $RUNNER" >&2
    exit 4
fi
if pgrep -af "[r]un_remnet_phase_interaction_gate.py" >/dev/null; then
    echo "[queue] phase gate is already running; refusing a duplicate" >&2
    exit 5
fi

mkdir -p experiments_mmecg
echo "[queue] $(date -Is) waiting for GPU ${GPU_INDEX}; required free memory=${MIN_FREE_MIB} MiB" | tee -a "$QUEUE_LOG"

while true; do
    gpu_row=$(nvidia-smi --id="$GPU_INDEX" \
        --query-gpu=memory.free,memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
    IFS=',' read -r free_mib used_mib util_pct <<< "$gpu_row"
    free_mib=${free_mib//[[:space:]]/}
    used_mib=${used_mib//[[:space:]]/}
    util_pct=${util_pct//[[:space:]]/}

    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
        echo "[queue] $(date -Is) GPU ready: free=${free_mib} MiB used=${used_mib} MiB util=${util_pct}%" | tee -a "$QUEUE_LOG"
        break
    fi

    echo "[queue] $(date -Is) waiting: free=${free_mib:-unknown} MiB used=${used_mib:-unknown} MiB util=${util_pct:-unknown}%" | tee -a "$QUEUE_LOG"
    sleep "$POLL_SECONDS"
done

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
echo "[queue] $(date -Is) launching $RUNNER" | tee -a "$QUEUE_LOG"
exec "$PYTHON_BIN" "$RUNNER" 2>&1 | tee -a "$RUN_LOG"
