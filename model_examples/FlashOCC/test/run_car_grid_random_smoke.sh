#!/usr/bin/env bash
# car_perception_grid 随机数据冒烟：2 路假图 + OM 分片 NPU 推理 + 分模块耗时
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate torch2.1.0_py38
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/customize:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/customize/op_api/lib:${CONDA_PREFIX}/lib/python3.8/site-packages/torch_npu/lib:${LD_LIBRARY_PATH:-}
export ASCEND_LAUNCH_BLOCKING=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASHOCC_ROOT="${FLASHOCC_ROOT:-${SCRIPT_DIR}/../FlashOCC}"
cd "${FLASHOCC_ROOT}"
export PYTHONPATH=$(pwd)/projects:$(pwd)/mmdetection3d:$(pwd):${PYTHONPATH:-}

CKPT="${CKPT:-work_dirs/car_grid_v4/epoch_12_ema.pth}"
SMOKE_PKL="data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_val_smoke.pkl"
MANIFEST="work_dirs/onnx_split_car_grid/flashocc_car_grid_deploy_manifest.json"
DEPLOY_CFG="projects/configs/flashocc/flashocc-r50-car-grid-trt.py"
LOG="work_dirs/car_grid_random_smoke.log"
PROFILE_OUT="work_dirs/onnx_split_car_grid/car_grid_profile_report.json"
PROFILE_WARMUP="${PROFILE_WARMUP:-2}"
PROFILE_ITERS="${PROFILE_ITERS:-3}"
PROFILE_DETAIL="${PROFILE_DETAIL:-0}"
PROFILE_ARGS=(--profile --profile-warmup "${PROFILE_WARMUP}" --profile-iters "${PROFILE_ITERS}" --profile-out "${PROFILE_OUT}")
if [[ "${PROFILE_DETAIL}" == "1" ]]; then
  PROFILE_ARGS+=(--profile-detail)
fi

for f in "${CKPT}" "${MANIFEST}" \
  work_dirs/onnx_split_car_grid/flashocc_car_grid_part1.om \
  work_dirs/onnx_split_car_grid/flashocc_car_grid_part3.om; do
  [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done

echo "========== [1/2] Generate random 2-cam smoke data =========="
python3 tools/create_car_grid_smoke_data.py

echo "========== [2/2] OM split infer + profile (checkpoint=${CKPT}) =========="
timeout -k 30s 45m python3 tools/run_split_infer_npu.py \
  "${DEPLOY_CFG}" \
  "${CKPT}" \
  "${MANIFEST}" \
  --fuse-conv-bn \
  --gpu-id 0 \
  --om-only \
  "${PROFILE_ARGS[@]}" \
  --cfg-options data.test.ann_file="${SMOKE_PKL}" data.val.ann_file="${SMOKE_PKL}" \
  2>&1 | tee "${LOG}"

code=${PIPESTATUS[0]}
echo "CAR_GRID_RANDOM_SMOKE_EXIT_CODE=${code}"
exit ${code}
