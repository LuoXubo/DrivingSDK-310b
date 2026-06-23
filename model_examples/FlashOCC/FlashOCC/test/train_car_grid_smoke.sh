#!/usr/bin/env bash
# Smoke train: car_perception_grid 2cam / 3cls / Dz2
set -euo pipefail

NUM_NPU=${1:-1}
BATCH_SIZE=${2:-1}
WORKDIR=${WORKDIR:-work_dirs/car_grid_smoke}

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

PKL_TRAIN="data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_train.pkl"
if [ -f "$PKL_TRAIN" ]; then
  echo "[1/4] Dataset pkl exists, skip conversion."
else
  echo "[1/4] Convert dataset from /data/car_perception_grid ..."
  python tools/create_car_perception_flashocc.py --src /data/car_perception_grid
fi

echo "[1b/4] Ensure media linked under nuscenes/samples (Docker paths) ..."
python tools/link_car_perception_media.py

echo "[2/4] Build 2-sample smoke pkl ..."
python - <<'PY'
import pickle
src = "data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_train.pkl"
dst = "data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_smoke.pkl"
with open(src, "rb") as f:
    data = pickle.load(f)
data["infos"] = data["infos"][:2]
with open(dst, "wb") as f:
    pickle.dump(data, f)
print("smoke samples:", len(data["infos"]))
PY

echo "[3/4] Train smoke: ${NUM_NPU} NPU, batch ${BATCH_SIZE} ..."
CFG=projects/configs/flashocc/flashocc-r50-car-grid.py
SMOKE_OPTS="data.train.ann_file=data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_smoke.pkl"

export PYTHONPATH="${ROOT}:${ROOT}/projects:${PYTHONPATH:-}"

# NPU adapt (idempotent)
sed -i 's/^from multiprocessing.dummy import Pool as ThreadPool/# from multiprocessing.dummy import Pool as ThreadPool/' projects/mmdet3d_plugin/models/detectors/bevdet_occ.py || true
sed -i 's/^from ...ops import nearest_assign/# from ...ops import nearest_assign/' projects/mmdet3d_plugin/models/detectors/bevdet_occ.py || true
sed -i 's/^\(\s*\)is_cuda\s*=\s*True/\1is_cuda = False/' projects/mmdet3d_plugin/models/detectors/bevdet_occ.py || true

if [ "$NUM_NPU" -gt 1 ]; then
  bash tools/dist_train.sh "$CFG" "$NUM_NPU" --work-dir "$WORKDIR" \
    --cfg-options data.samples_per_gpu="$BATCH_SIZE" runner.max_epochs=1 data.workers_per_gpu=0 "$SMOKE_OPTS"
else
  python tools/train.py "$CFG" --work-dir "$WORKDIR" --gpu-id 0 \
    --cfg-options data.samples_per_gpu="$BATCH_SIZE" runner.max_epochs=1 data.workers_per_gpu=0 "$SMOKE_OPTS"
fi

echo "[4/4] Done. work_dir=$WORKDIR"
