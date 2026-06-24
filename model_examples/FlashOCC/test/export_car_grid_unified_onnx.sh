#!/usr/bin/env bash
# Export part1+part2(BEVPoolV3)+part3 ONNX, merge e2e, generate ATC script for single OM.
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate torch2.1.0_py38

FLASHOCC_ROOT="${FLASHOCC_ROOT:-/root/flashocc_dir/DrivingSDK/model_examples/FlashOCC/FlashOCC}"
cd "${FLASHOCC_ROOT}"
export PYTHONPATH=$(pwd)/projects:$(pwd)/mmdetection3d:$(pwd):${PYTHONPATH:-}

CKPT="${CKPT:-work_dirs/car_grid_v4/epoch_12_ema.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/onnx_unified_car_grid}"
PREFIX="${PREFIX:-flashocc_car_grid}"
DEPLOY_CFG="${DEPLOY_CFG:-projects/configs/flashocc/flashocc-r50-car-grid-trt.py}"
SMOKE_PKL="${SMOKE_PKL:-data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_val_smoke.pkl}"
SOC="${SOC_VERSION:-Ascend310B1}"

python3 tools/create_car_grid_smoke_data.py 2>/dev/null || true

python3 tools/export_onnx_unified_npu.py \
  "${DEPLOY_CFG}" \
  "${CKPT}" \
  "${WORK_DIR}" \
  --prefix "${PREFIX}" \
  --fuse-conv-bn \
  --soc-version "${SOC}" \
  --cfg-options data.test.ann_file="${SMOKE_PKL}" data.val.ann_file="${SMOKE_PKL}"

echo "ONNX export done. Next (requires cust_onnx_parsers.so with BEVPoolV3):"
echo "  bash ${WORK_DIR}/atc_convert_${PREFIX}_e2e.sh"
