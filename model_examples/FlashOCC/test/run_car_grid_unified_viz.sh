#!/usr/bin/env bash
# 使用 merged OM 对自定义数据推理，保存栅格图/GT/输入图可视化及耗时 profile
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

DEPLOY_CFG="${DEPLOY_CFG:-projects/configs/flashocc/flashocc-r50-car-grid-trt.py}"
MANIFEST="${MANIFEST:-work_dirs/onnx_unified_car_grid/flashocc_car_grid_unified_deploy_manifest.json}"
OM_PATH="${OM_PATH:-work_dirs/onnx_unified_car_grid/flashocc_car_grid_merged.om}"
OUT_DIR="${OUT_DIR:-work_dirs/unified_viz_results}"
TEST_PKL="${TEST_PKL:-data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_test10.pkl}"
SAMPLES="${SAMPLES:-0}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"

python3 tools/run_unified_viz_npu.py \
  "${DEPLOY_CFG}" \
  "${MANIFEST}" \
  --om-path "${OM_PATH}" \
  --out-dir "${OUT_DIR}" \
  --sample-idx "${SAMPLE_IDX}" \
  --samples "${SAMPLES}" \
  --profile \
  --cfg-options \
    data.test.ann_file="${TEST_PKL}" \
    data.val.ann_file="${TEST_PKL}" \
    data.test.data_root=data/car_perception_grid/nuscenes/ \
    data.val.data_root=data/car_perception_grid/nuscenes/

echo "VIZ_DONE out_dir=${OUT_DIR}"
