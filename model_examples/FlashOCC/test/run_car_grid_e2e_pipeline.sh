#!/usr/bin/env bash
# car_grid: export part1/2/3 ONNX -> merge -> ATC e2e OM -> test10 推理
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

CKPT="${CKPT:-work_dirs/car_grid_v4/epoch_12_ema.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/onnx_unified_car_grid}"
PREFIX="${PREFIX:-flashocc_car_grid}"
DEPLOY_CFG="${DEPLOY_CFG:-projects/configs/flashocc/flashocc-r50-car-grid-trt.py}"
TEST_PKL="${TEST_PKL:-data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_test10.pkl}"
SOC="${SOC_VERSION:-Ascend310B1}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
LOG="${WORK_DIR}/e2e_pipeline.log"

mkdir -p "${WORK_DIR}"

echo "========== [1/4] Export part1/part2/part3 ONNX (sample_idx=${SAMPLE_IDX}) =========="
python3 tools/export_onnx_unified_npu.py \
  "${DEPLOY_CFG}" \
  "${CKPT}" \
  "${WORK_DIR}" \
  --prefix "${PREFIX}" \
  --fuse-conv-bn \
  --soc-version "${SOC}" \
  --sample-idx "${SAMPLE_IDX}" \
  --skip-e2e \
  --part2-std-ops \
  --cfg-options data.test.ann_file="${TEST_PKL}" data.val.ann_file="${TEST_PKL}" \
  2>&1 | tee -a "${LOG}"

echo "========== [2/4] Merge part1+part2+part3 -> single ONNX =========="
python3 tools/merge_onnx_flashocc.py \
  "${WORK_DIR}/${PREFIX}_part1.onnx" \
  "${WORK_DIR}/${PREFIX}_part2.onnx" \
  "${WORK_DIR}/${PREFIX}_part3.onnx" \
  "${WORK_DIR}/${PREFIX}_merged.onnx" \
  2>&1 | tee -a "${LOG}"

echo "========== [3/4] ATC: merged ONNX -> single OM =========="
E2E_OM="${WORK_DIR}/${PREFIX}_merged"
atc --model="${WORK_DIR}/${PREFIX}_merged.onnx" \
  --framework=5 \
  --output="${E2E_OM}" \
  --input_format=NCHW \
  --input_shape="img:2,3,256,704" \
  --soc_version="${SOC}" \
  --precision_mode=allow_fp32_to_fp16 \
  2>&1 | tee -a "${LOG}"

echo "========== [4/4] NPU infer on test10 (${TEST_PKL}) =========="
python3 tools/run_unified_infer_npu.py \
  "${DEPLOY_CFG}" \
  "${WORK_DIR}/${PREFIX}_unified_deploy_manifest.json" \
  --om-path "${E2E_OM}.om" \
  --samples 10 \
  --profile \
  --cfg-options data.test.ann_file="${TEST_PKL}" data.val.ann_file="${TEST_PKL}" \
  2>&1 | tee -a "${LOG}"

echo "E2E_PIPELINE_OK merged_om=${E2E_OM}.om log=${LOG}"
