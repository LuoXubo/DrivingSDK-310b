#!/usr/bin/env bash
# car_perception_grid test10 推理：优先 merged OM，否则 split OM + bev_pool_v3
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate torch2.1.0_py38
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/customize:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=${ASCEND_OPP_PATH}/vendors/customize/op_api/lib:${CONDA_PREFIX}/lib/python3.8/site-packages/torch_npu/lib:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

FLASHOCC_ROOT="${FLASHOCC_ROOT:-/root/flashocc_dir/DrivingSDK-310b/model_examples/FlashOCC/FlashOCC}"
cd "${FLASHOCC_ROOT}"
export PYTHONPATH=$(pwd)/projects:$(pwd)/mmdetection3d:$(pwd):${PYTHONPATH:-}

TEST_PKL="data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_test10.pkl"
DEPLOY_CFG="projects/configs/flashocc/flashocc-r50-car-grid-trt.py"
CKPT="work_dirs/car_grid_v4/epoch_12_ema.pth"
MERGED_OM="work_dirs/onnx_unified_car_grid/flashocc_car_grid_merged.om"
SPLIT_MANIFEST="work_dirs/onnx_split_car_grid/flashocc_car_grid_deploy_manifest.json"
UNIFIED_MANIFEST="work_dirs/onnx_unified_car_grid/flashocc_car_grid_unified_deploy_manifest.json"
LOG="work_dirs/onnx_unified_car_grid/test10_infer.log"

mkdir -p work_dirs/onnx_unified_car_grid

if [[ -f "${MERGED_OM}" ]]; then
  echo "========== Unified merged OM (test10 x10) =========="
  python3 tools/run_unified_infer_npu.py \
    "${DEPLOY_CFG}" \
    "${UNIFIED_MANIFEST}" \
    --om-path "${MERGED_OM}" \
    --samples 10 \
    --profile \
    --cfg-options data.test.ann_file="${TEST_PKL}" data.val.ann_file="${TEST_PKL}" \
    2>&1 | tee "${LOG}"
else
  echo "Merged OM not found (${MERGED_OM}); using split OM + bev_pool_v3"
  python3 tools/run_split_infer_npu.py \
    "${DEPLOY_CFG}" \
    "${CKPT}" \
    "${SPLIT_MANIFEST}" \
    --fuse-conv-bn \
    --gpu-id 0 \
    --om-only \
    --samples 10 \
    --profile \
    --profile-warmup 1 \
    --profile-iters 3 \
    --profile-out work_dirs/onnx_unified_car_grid/test10_split_profile.json \
    --cfg-options data.test.ann_file="${TEST_PKL}" data.val.ann_file="${TEST_PKL}" \
    2>&1 | tee "${LOG}"
fi

echo "TEST10_INFER_DONE log=${LOG}"
