#!/usr/bin/env bash
# Wait for a genuinely free GPU, smoke L1/L2/L3, then run the registered 3x3 gate.
set -euo pipefail

ROOT=/home/qhh2237/Projects/BeatAware-Radar2ECGNet
PY=/home/qhh2237/miniconda3/envs/cyberbrain/bin/python
TAG=q1_continuous_ecg_loss_ablation
LOG_DIR="$ROOT/experiments_mmecg/$TAG"
MIN_FREE_MIB=12000
mkdir -p "$LOG_DIR"
cd "$ROOT"

exec >>"$LOG_DIR/queue.log" 2>&1
echo "[$(date --iso-8601=seconds)] queue started; waiting for ${MIN_FREE_MIB} MiB free"

while true; do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
    break
  fi
  echo "[$(date --iso-8601=seconds)] GPU busy; free=${free_mib:-unknown} MiB"
  sleep 30
done

echo "[$(date --iso-8601=seconds)] GPU available; starting isolated smoke runs"
for loss_arm in L1 L2 L3; do
  "$PY" scripts/train_continuous_ecg_pilot.py \
    --arm B1 --loss_arm "$loss_arm" --fold 1 --smoke \
    --exp_tag "$TAG" --device cuda:0
done

echo "[$(date --iso-8601=seconds)] smoke passed; starting nine formal runs"
for fold in 1 5 9; do
  for loss_arm in L1 L2 L3; do
    "$PY" scripts/train_continuous_ecg_pilot.py \
      --arm B1 --loss_arm "$loss_arm" --fold "$fold" \
      --exp_tag "$TAG" --device cuda:0
  done
done

echo "[$(date --iso-8601=seconds)] all nine formal runs completed"
