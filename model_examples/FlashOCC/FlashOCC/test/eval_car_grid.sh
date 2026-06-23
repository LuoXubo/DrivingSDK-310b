#!/usr/bin/env bash
# Evaluate car_perception_grid checkpoint and visualize pred vs GT.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT=${1:-work_dirs/car_grid_2p_npu67/epoch_24_ema.pth}
OUT_DIR=${2:-work_dirs/car_grid_test_results}
GPU_ID=${GPU_ID:-0}

export PYTHONPATH="${ROOT}/projects:${ROOT}/mmdetection3d:${ROOT}:${PYTHONPATH:-}"

python tools/eval_car_grid_occ.py \
  --checkpoint "$CKPT" \
  --out-dir "$OUT_DIR" \
  --gpu-id "$GPU_ID"
