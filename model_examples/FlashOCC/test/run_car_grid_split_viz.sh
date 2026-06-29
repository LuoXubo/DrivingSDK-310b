#!/usr/bin/env bash
# Split deploy Torch NPU inference + visualization on test10 (distinct from merged OM viz).
set -euo pipefail

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate torch2.1.0_py38
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib:${CONDA_PREFIX}/lib/python3.8/site-packages/torch_npu/lib:${LD_LIBRARY_PATH:-}
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

FLASHOCC_ROOT="${FLASHOCC_ROOT:-/root/flashocc_dir/DrivingSDK-310b/model_examples/FlashOCC/FlashOCC}"
cd "${FLASHOCC_ROOT}"
export PYTHONPATH=$(pwd)/projects:$(pwd)/mmdetection3d:$(pwd):${PYTHONPATH:-}

CKPT="${CKPT:-work_dirs/car_grid_v4/epoch_12_ema.pth}"
OUT_DIR="${OUT_DIR:-work_dirs/test10_torch_npu_viz}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-work_dirs/onnx_split_car_grid_eval/flashocc_car_grid_deploy_manifest.json}"
SAMPLES="${SAMPLES:-0}"

python3 tools/run_split_viz_npu.py \
  --checkpoint "${CKPT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --samples "${SAMPLES}" \
  --profile

echo "TORCH_NPU_VIZ_DONE out_dir=${OUT_DIR}"
