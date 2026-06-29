#!/usr/bin/env bash
# Re-export split part1/part3 ONNX with eval-aligned (B,N,3,H,W) image_encoder path.
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate torch2.1.0_py38

FLASHOCC_ROOT="${FLASHOCC_ROOT:-/root/flashocc_dir/DrivingSDK-310b/model_examples/FlashOCC/FlashOCC}"
cd "${FLASHOCC_ROOT}"
export PYTHONPATH=$(pwd)/projects:$(pwd)/mmdetection3d:$(pwd):${PYTHONPATH:-}

CKPT="${CKPT:-work_dirs/car_grid_v4/epoch_12_ema.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/onnx_split_car_grid_eval}"
PREFIX="${PREFIX:-flashocc_car_grid}"
DEPLOY_CFG="${DEPLOY_CFG:-projects/configs/flashocc/flashocc-r50-car-grid-trt.py}"
TEST10_PKL="${TEST10_PKL:-data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_test10.pkl}"

SOC="${SOC_VERSION:-Ascend310B1}"

python3 tools/export_onnx_split_npu.py \
  "${DEPLOY_CFG}" \
  "${CKPT}" \
  "${WORK_DIR}" \
  --prefix "${PREFIX}" \
  --fuse-conv-bn \
  --part1-layout eval \
  --export-on-cpu \
  --soc-version "${SOC}" \
  --no-parallel-atc \
  --cfg-options data.test.ann_file="${TEST10_PKL}" data.val.ann_file="${TEST10_PKL}"

echo "Eval-aligned split ONNX export done."
echo "  manifest: ${WORK_DIR}/${PREFIX}_deploy_manifest.json"
echo "  Next: bash ${WORK_DIR}/atc_convert_${PREFIX}.sh"
