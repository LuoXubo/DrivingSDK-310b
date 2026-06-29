#!/usr/bin/env bash
# Export eval-aligned part1 + segment-sum part2 + eval-aligned part3, merge e2e ONNX.
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate torch2.1.0_py38

FLASHOCC_ROOT="${FLASHOCC_ROOT:-/root/flashocc_dir/DrivingSDK-310b/model_examples/FlashOCC/FlashOCC}"
cd "${FLASHOCC_ROOT}"
export PYTHONPATH=$(pwd)/projects:$(pwd)/mmdetection3d:$(pwd):${PYTHONPATH:-}

CKPT="${CKPT:-work_dirs/car_grid_v4/epoch_12_ema.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/onnx_unified_car_grid}"
PREFIX="${PREFIX:-flashocc_car_grid}"
DEPLOY_CFG="${DEPLOY_CFG:-projects/configs/flashocc/flashocc-r50-car-grid-trt.py}"
TEST10_PKL="${TEST10_PKL:-data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_test10.pkl}"
SOC="${SOC_VERSION:-Ascend310B1}"

python3 tools/export_onnx_unified_npu.py \
  "${DEPLOY_CFG}" \
  "${CKPT}" \
  "${WORK_DIR}" \
  --prefix "${PREFIX}" \
  --fuse-conv-bn \
  --part1-layout eval \
  --part2-segment-sum \
  --atc-part1-precision force_fp32 \
  --atc-merged-precision force_fp32 \
  --soc-version "${SOC}" \
  --cfg-options data.test.ann_file="${TEST10_PKL}" data.val.ann_file="${TEST10_PKL}"

echo "Merged segment-sum ONNX export done."
echo "  merged: ${WORK_DIR}/${PREFIX}_merged.onnx"
echo "  Next: bash ${WORK_DIR}/atc_convert_${PREFIX}_merged_segment_sum.sh"
